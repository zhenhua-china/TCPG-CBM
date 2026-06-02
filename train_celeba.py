import os
os.environ.setdefault('TIMM_USE_HF', '0')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sklearn.metrics
from model import TCPG_CBM
from extractors import ViT_PatchExtractor_TCPA, ViT_PatchExtractor, CLIP_ConceptEncoder
from data.celeba import CELEBA_CONFIG, CONCEPT_SEMANTICS, generate_data




def collate_from_celeba_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2 and torch.is_tensor(batch[0]):
            images = batch[0]
            target = batch[1]
            if isinstance(target, (list, tuple)) and len(target) == 2:
                labels = target[0].long()
                concepts = target[1].float()
            else:
                labels = target.long()
                concepts = None
            return images, labels, concepts
        if len(batch) == 3 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]) and torch.is_tensor(batch[2]):
            images = batch[0]
            labels = batch[1].long()
            concepts = batch[2].float()
            return images, labels, concepts
        images = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1][0] for b in batch]).long()
        concepts = torch.stack([b[1][1] for b in batch]).float()
        return images, labels, concepts


def train_epoch(model, vit, clip_enc, loader, opt, crit_cls, crit_conc, device, l_orth, l_conc, l_reg, tune_tcpa=False, criterion_center=None, use_beyond_losses=True, l_conc_extra=0.1, l_equiv=0.1, l_center=0.1):
    model.train()
    vit.eval()
    if tune_tcpa:
        vit.train()
    total_loss = 0.0
    class_loss_sum = 0.0
    concept_loss_sum = 0.0
    orth_loss_sum = 0.0
    reg_loss_sum = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        images, y, c = collate_from_celeba_batch(batch)
        images = images.to(device)
        y = y.to(device)
        c = c.to(device)
        bsz = images.size(0)
        if tune_tcpa:
            patch = vit(images)
            concept_feat = clip_enc(bsz)
        else:
            with torch.no_grad():
                patch = vit(images)
                concept_feat = clip_enc(bsz)
        logits, act, orth = model(patch, concept_feat)
        conc_extra = torch.tensor(0.0, device=device)
        equiv_extra = torch.tensor(0.0, device=device)
        center_extra = torch.tensor(0.0, device=device)
        if use_beyond_losses and getattr(model, "use_parts", False):
            z_parts, w_parts, s_fg, maps = model.get_parts(patch)
            B = images.size(0)
            if maps.shape[-1] > 1:
                from lib import landmark_coordinates, batch_rigid_transform
                from losses import conc_loss
                maps_parts = maps[:, :-1, :, :]
                loc_x, loc_y, grid_x, grid_y = landmark_coordinates(maps_parts, device)
                conc_extra = conc_loss(loc_x, loc_y, grid_x, grid_y, maps_parts) * l_conc_extra
                angle = torch.rand(B).to(device) * 180 - 90
                translate = torch.rand(B, 2).to(device) * 0.2 - 0.1
                scale = torch.rand(B).to(device) * 0.5 + 0.9
                transf_img, _ = batch_rigid_transform(images, angle.cpu().numpy(), translate.cpu().numpy(), scale=scale.cpu().numpy(), invert=False, device=device)
                if tune_tcpa:
                    patch_t = vit(transf_img.to(device))
                else:
                    with torch.no_grad():
                        patch_t = vit(transf_img.to(device))
                _, _, _, maps_t = model.get_parts(patch_t)
                rot_back, maskI = batch_rigid_transform(maps_t, angle.cpu().numpy(), translate.cpu().numpy(), scale=scale.cpu().numpy(), invert=True, device=device)
                maskI = maskI[:, :1]
                import torch.nn.functional as Fnn
                cos_sim = Fnn.cosine_similarity((maps_parts * maskI).reshape(B, maps_parts.shape[1], -1), (rot_back[:, :-1, :, :] * maskI).reshape(B, maps_parts.shape[1], -1), dim=-1)
                equiv_extra = (1.0 - cos_sim.mean()) * l_equiv
            if criterion_center is not None and w_parts.shape[1] > 0:
                center_extra = criterion_center(z_parts, y) * l_center
        class_logits_detached = model.concept_to_class(act.detach())
        loss_cls = crit_cls(class_logits_detached.float(), y)
        loss_conc = crit_conc(act, c.to(dtype=act.dtype))
        reg_dict = model.concept_to_class.reg_loss() if hasattr(model.concept_to_class, "reg_loss") else {}
        reg_sum = sum(reg_dict.values()) if len(reg_dict) > 0 else torch.tensor(0.0, device=device)
        loss = loss_cls + l_conc * loss_conc + l_orth * orth + l_reg * reg_sum + conc_extra + equiv_extra + center_extra
        opt.zero_grad()
        loss.backward()
        if tune_tcpa and hasattr(vit, "tcpas"):
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(vit.tcpas.parameters()), max_norm=1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total_loss += float(loss.item())
        class_loss_sum += float(loss_cls.item())
        concept_loss_sum += float(loss_conc.item())
        orth_loss_sum += float(orth.item())
        reg_loss_sum += float(reg_sum.item())
        _, pred = torch.max(logits, 1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
        pbar.set_postfix(loss=total_loss / (total // bsz + 1), acc=100.0 * correct / max(total, 1))
    return {
        'loss': total_loss / len(loader),
        'class_loss': class_loss_sum / len(loader),
        'concept_loss': concept_loss_sum / len(loader),
        'orth_loss': orth_loss_sum / len(loader),
        'reg_loss': reg_loss_sum / len(loader),
        'accuracy': 100.0 * correct / max(total, 1)
    }


@torch.no_grad()
def evaluate(model, vit, clip_enc, loader, crit_cls, crit_conc, device, l_orth, l_conc, l_reg):
    model.eval()
    vit.eval()
    total_loss = 0.0
    class_loss_sum = 0.0
    concept_loss_sum = 0.0
    reg_loss_sum = 0.0
    correct = 0
    total = 0
    concept_preds_list = []
    concept_labels_list = []
    pbar = tqdm(loader, desc='Evaluating')
    for batch in pbar:
        images, y, c = collate_from_celeba_batch(batch)
        images = images.to(device)
        y = y.to(device)
        c = c.to(device)
        bsz = images.size(0)
        patch = vit(images)
        concept_feat = clip_enc(bsz)
        logits, act, orth = model(patch, concept_feat)
        loss_cls = crit_cls(logits.float(), y)
        loss_conc = crit_conc(act, c.to(dtype=act.dtype))
        reg_dict = model.concept_to_class.reg_loss() if hasattr(model.concept_to_class, "reg_loss") else {}
        reg_sum = sum(reg_dict.values()) if len(reg_dict) > 0 else torch.tensor(0.0, device=device)
        loss = loss_cls + l_conc * loss_conc + l_orth * orth + l_reg * reg_sum
        total_loss += float(loss.item())
        class_loss_sum += float(loss_cls.item())
        concept_loss_sum += float(loss_conc.item())
        reg_loss_sum += float(reg_sum.item())
        _, pred = torch.max(logits, 1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
        concept_preds_list.append((act >= 0.5).float().cpu())
        concept_labels_list.append(c.cpu())
    concept_preds = torch.cat(concept_preds_list, dim=0).numpy()
    concept_labels = torch.cat(concept_labels_list, dim=0).numpy()
    concept_accuracy = (concept_preds == concept_labels).mean()
    concept_f1_macro = sklearn.metrics.f1_score(concept_labels.reshape(-1), concept_preds.reshape(-1), average='macro')
    concept_overall_accuracy = sklearn.metrics.accuracy_score(concept_labels, concept_preds)
    return {
        'loss': total_loss / len(loader),
        'class_loss': class_loss_sum / len(loader),
        'concept_loss': concept_loss_sum / len(loader),
        'reg_loss': reg_loss_sum / len(loader),
        'accuracy': 100.0 * correct / max(total, 1),
        'concept_accuracy': concept_accuracy * 100.0,
        'concept_f1_macro': concept_f1_macro * 100.0,
        'concept_overall_accuracy': concept_overall_accuracy * 100.0
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output_dir, f'dotcbm_celeba_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    writer = SummaryWriter(os.path.join(out_dir, 'logs'))
    train_dl, val_dl, test_dl, _, meta = generate_data(args.data_dir, resol=224, batch_size=args.batch_size, num_workers=args.num_workers, config=CELEBA_CONFIG, output_dataset_vars=True)
    concept_names = meta[3]
    n_classes = CELEBA_CONFIG.get('num_classes', meta[1])
    print(f"训练集大小: {len(train_dl.dataset)} | 验证集大小: {len(val_dl.dataset)} | 测试集大小: {len(test_dl.dataset)}", flush=True)
    print(f"类别数: {n_classes} | 概念数: {len(concept_names)}", flush=True)
    if args.tune_tcpa:
        vit = ViT_PatchExtractor_TCPA(
            model_name=args.vit_model,
            weights_path=(getattr(args, 'vit_weights_path', None) or None),
            pretrained=False,
            freeze_backbone=True,
            inject='last_k',
            last_k=args.tcpa_last_k,
            custom_layers=args.tcpa_layers,
            n_img_prompts=args.tcpa_n_img_prompts,
            n_cls_prompts=args.tcpa_n_cls_prompts,
            topk=args.tcpa_topk,
            tau=args.tcpa_tau,
            dropout=args.tcpa_dropout,
            use_mlp=(not args.tcpa_no_mlp),
        ).to(device)
    else:
        vit = ViT_PatchExtractor(
            model_name=args.vit_model,
            weights_path=(getattr(args, 'vit_weights_path', None) or None),
            pretrained=False,
            freeze=True
        ).to(device)
    clip_enc = CLIP_ConceptEncoder(
        concept_list=concept_names,
        model_name=args.clip_model,
        device=device,
        pretrained_path=args.clip_pretrained
    )
    model = TCPG_CBM(
        num_patches=196,
        num_concepts=len(concept_names),
        num_classes=n_classes,
        patch_dim=vit.embed_dim,
        concept_dim=clip_enc.embed_dim,
        hidden_dim=args.hidden_dim,
        ot_reg=args.ot_reg,
        dropout=args.dropout,
        activation_method=args.activation_method,
        activation_tau=(args.activation_tau if args.activation_tau is not None else args.ot_reg),
        use_parts=args.use_parts,
        num_parts=args.num_parts,
        part_tv_weight=args.part_tv_weight,
        part_distinct_weight=args.part_distinct_weight,
        part_presence_weight=args.part_presence_weight,
        concept_head=getattr(args, "concept_head", "nam")
    ).to(device)
    crit_cls = nn.CrossEntropyLoss()
    crit_conc = nn.BCELoss()
    criterion_center = None
    if args.use_parts and device.type == 'cuda':
        try:
            from losses import PartCenterLoss
            criterion_center = PartCenterLoss(num_classes=n_classes, num_parts=args.num_parts, feat_dim=args.hidden_dim, use_gpu=(device.type == 'cuda'))
        except Exception:
            criterion_center = None
    if args.tune_tcpa:
        opt = optim.Adam([
            {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': vit.tcpas.parameters(), 'lr': args.lr_tcpa, 'weight_decay': args.weight_decay}
        ])
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    best_val_class_acc = 0.0
    best_val_concept_acc = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        tr = train_epoch(model, vit, clip_enc, train_dl, opt, crit_cls, crit_conc, device, args.lambda_orth, args.lambda_concept, args.lambda_reg, tune_tcpa=args.tune_tcpa, criterion_center=criterion_center, use_beyond_losses=True, l_conc_extra=args.l_conc, l_equiv=args.l_equiv, l_center=args.l_center)
        va = evaluate(model, vit, clip_enc, val_dl, crit_cls, crit_conc, device, args.lambda_orth, args.lambda_concept, args.lambda_reg)
        scheduler.step()
        writer.add_scalar('train/loss', tr['loss'], epoch)
        writer.add_scalar('train/acc', tr['accuracy'], epoch)
        writer.add_scalar('train/class_loss', tr['class_loss'], epoch)
        writer.add_scalar('train/concept_loss', tr['concept_loss'], epoch)
        writer.add_scalar('train/orth_loss', tr['orth_loss'], epoch)
        writer.add_scalar('train/lr', opt.param_groups[0]['lr'], epoch)
        writer.add_scalar('val/loss', va['loss'], epoch)
        writer.add_scalar('val/acc', va['accuracy'], epoch)
        writer.add_scalar('val/concept_acc', va['concept_accuracy'], epoch)
        writer.add_scalar('val/concept_f1', va['concept_f1_macro'], epoch)
        writer.add_scalar('val/concept_overall_acc', va.get('concept_overall_accuracy', 0.0), epoch)
        print(f"Validation - Top1 Acc: {va['accuracy']:.2f}%, Concept Acc: {va['concept_accuracy']:.2f}%, Concept Overall Acc: {va.get('concept_overall_accuracy', 0.0):.2f}%, Concept F1(macro): {va['concept_f1_macro']:.2f}%")
        if va['accuracy'] > best_val_class_acc:
            best_val_class_acc = va['accuracy']
            payload = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_class_acc': va['accuracy'], 'val_concept_acc': va['concept_accuracy']}
            if hasattr(vit, 'tcpas'):
                payload['tcpas_state_dict'] = vit.tcpas.state_dict()
                torch.save({'epoch': epoch, 'tcpas_state_dict': vit.tcpas.state_dict(), 'val_class_acc': va['accuracy'], 'val_concept_acc': va['concept_accuracy']}, os.path.join(out_dir, 'best_tcpa_class.pth'))
            torch.save(payload, os.path.join(out_dir, 'best_class_model.pth'))
            print(f"保存最佳分类权重，轮次 {epoch+1}：best_class_model.pth{', best_tcpa_class.pth' if hasattr(vit, 'tcpas') else ''}")
        if va['concept_accuracy'] > best_val_concept_acc:
            best_val_concept_acc = va['concept_accuracy']
            payload_c = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_class_acc': va['accuracy'], 'val_concept_acc': va['concept_accuracy']}
            if hasattr(vit, 'tcpas'):
                payload_c['tcpas_state_dict'] = vit.tcpas.state_dict()
                torch.save({'epoch': epoch, 'tcpas_state_dict': vit.tcpas.state_dict(), 'val_class_acc': va['accuracy'], 'val_concept_acc': va['concept_accuracy']}, os.path.join(out_dir, 'best_tcpa_concept.pth'))
            torch.save(payload_c, os.path.join(out_dir, 'best_concept_model.pth'))
            print(f"保存最佳概念权重，轮次 {epoch+1}：best_concept_model.pth{', best_tcpa_concept.pth' if hasattr(vit, 'tcpas') else ''}")
    best_class_path = os.path.join(out_dir, 'best_class_model.pth')
    test_metrics_mixed = None
    test_metrics_file = None
    if os.path.exists(best_class_path):
        ckpt = torch.load(best_class_path)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'epoch' in ckpt:
            try:
                print(f"使用验证集最佳分类权重进行测试（模型轮次 {int(ckpt['epoch']) + 1}）", flush=True)
            except Exception:
                print("使用验证集最佳分类权重进行测试（模型轮次未知）", flush=True)
        if hasattr(vit, 'tcpas'):
            if 'tcpas_state_dict' in ckpt or 'tcpa_state_dict' in ckpt:
                state_dict_key = 'tcpas_state_dict' if 'tcpas_state_dict' in ckpt else 'tcpa_state_dict'
                vit.tcpas.load_state_dict(ckpt[state_dict_key], strict=False)
                test_metrics_mixed = evaluate(model, vit, clip_enc, test_dl, crit_cls, crit_conc, device, args.lambda_orth, args.lambda_concept, args.lambda_reg)
            tcpa_path = os.path.join(out_dir, 'best_tcpa_class.pth')
            if os.path.exists(tcpa_path):
                tcpa_obj = torch.load(tcpa_path)
                if isinstance(tcpa_obj, dict) and ('tcpas_state_dict' in tcpa_obj or 'tcpa_state_dict' in tcpa_obj):
                    vit.tcpas.load_state_dict(tcpa_obj.get('tcpas_state_dict', tcpa_obj.get('tcpa_state_dict')), strict=False)
                    if 'epoch' in tcpa_obj:
                        try:
                            print(f"TCPA 使用验证集最佳分类对应的提示权重（轮次 {int(tcpa_obj['epoch']) + 1}）", flush=True)
                        except Exception:
                            print("TCPA 使用验证集最佳分类对应的提示权重（轮次未知）", flush=True)
                else:
                    vit.tcpas.load_state_dict(tcpa_obj, strict=False)
                test_metrics_file = evaluate(model, vit, clip_enc, test_dl, crit_cls, crit_conc, device, args.lambda_orth, args.lambda_concept, args.lambda_reg)
    final_metrics = test_metrics_file or test_metrics_mixed
    if final_metrics is None:
        final_metrics = evaluate(model, vit, clip_enc, test_dl, crit_cls, crit_conc, device, args.lambda_orth, args.lambda_concept, args.lambda_reg)
    print(f"Test  - Top1 Acc: {final_metrics['accuracy']:.2f}%, Concept Acc: {final_metrics['concept_accuracy']:.2f}%, Concept Overall Acc: {final_metrics.get('concept_overall_accuracy', 0.0):.2f}%, Concept F1(macro): {final_metrics['concept_f1_macro']:.2f}%", flush=True)
    if test_metrics_mixed is not None:
        print(f"Test (mixed TCPA)  - Top1 Acc: {test_metrics_mixed['accuracy']:.2f}%, Concept Acc: {test_metrics_mixed['concept_accuracy']:.2f}%, Concept Overall Acc: {test_metrics_mixed.get('concept_overall_accuracy', 0.0):.2f}%, Concept F1(macro): {test_metrics_mixed['concept_f1_macro']:.2f}%", flush=True)
    if test_metrics_file is not None:
        print(f"Test (file TCPA)   - Top1 Acc: {test_metrics_file['accuracy']:.2f}%, Concept Acc: {test_metrics_file['concept_accuracy']:.2f}%, Concept Overall Acc: {test_metrics_file.get('concept_overall_accuracy', 0.0):.2f}%, Concept F1(macro): {test_metrics_file['concept_f1_macro']:.2f}%", flush=True)
    writer.add_scalar('test/acc', final_metrics['accuracy'], args.epochs)
    writer.add_scalar('test/concept_acc', final_metrics['concept_accuracy'], args.epochs)
    writer.add_scalar('test/concept_overall_acc', final_metrics.get('concept_overall_accuracy', 0.0), args.epochs)
    writer.add_scalar('test/concept_f1_macro', final_metrics['concept_f1_macro'], args.epochs)
    with open(os.path.join(out_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TCPG-CBM on CelebA')
    parser.add_argument('--data_dir', type=str, required=False, default='')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--vit_model', type=str, default='vit_base_patch16_224')
    parser.add_argument('--clip_model', type=str, default='ViT-B-32')
    parser.add_argument('--clip_pretrained', type=str, required=False, default='')
    parser.add_argument('--vit_weights_path', type=str, required=False, default='')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--ot_reg', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation_method', type=str, default='weighted')
    parser.add_argument('--activation_tau', type=float, default=1)
    parser.add_argument('--use_parts', action='store_true', default=True)
    parser.add_argument('--num_parts', type=int, default=9)
    parser.add_argument('--part_tv_weight', type=float, default=0.1)
    parser.add_argument('--part_distinct_weight', type=float, default=0.1)
    parser.add_argument('--part_presence_weight', type=float, default=0.1)
    parser.add_argument('--lambda_orth', type=float, default=0.1)
    parser.add_argument('--lambda_concept', type=float, default=5.0)
    parser.add_argument('--lambda_reg', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--tune_tcpa', action='store_true', default=True)
    parser.add_argument('--lr_tcpa', type=float, default=3e-4)
    parser.add_argument('--tcpa_last_k', type=int, default=12)
    parser.add_argument('--tcpa_layers', type=int, nargs='*')
    parser.add_argument('--tcpa_n_img_prompts', type=int, default=20)
    parser.add_argument('--tcpa_n_cls_prompts', type=int, default=4)
    parser.add_argument('--tcpa_topk', type=int, default=4)
    parser.add_argument('--tcpa_tau', type=float, default=0.1)
    parser.add_argument('--tcpa_dropout', type=float, default=0.1)
    parser.add_argument('--tcpa_no_mlp', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--l_conc', type=float, default=0.1)
    parser.add_argument('--l_equiv', type=float, default=0.1)
    parser.add_argument('--l_center', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--concept_head', type=str, choices=['nam', 'mlp'], default='mlp')
    args = parser.parse_args()
    main(args)
