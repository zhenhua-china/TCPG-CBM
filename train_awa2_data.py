import os
os.environ.setdefault('TIMM_USE_HF', '0')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np
import sklearn.metrics

from model import DOT_CBM
from extractors import ViT_PatchExtractor, CLIP_ConceptEncoder, ViT_PatchExtractor_TCPA
from data.data_util import get_dataset


def train_epoch(
    model: DOT_CBM,
    vit_extractor,
    clip_encoder,
    train_loader,
    optimizer: optim.Optimizer,
    criterion_class: nn.Module,
    criterion_concept: nn.Module,
    criterion_center: Optional[nn.Module],
    device: torch.device,
    lambda_orth: float = 0.1,
    lambda_concept: float = 1.0,
    lambda_reg: float = 1.0,
    use_priors=False,
    tune_tcpa: bool = False,
    use_beyond_losses: bool = True,
    l_conc: float = 0.1,
    l_equiv: float = 0.1,
    l_center: float = 0.1
) -> dict:
    model.train()
    vit_extractor.eval()
    if tune_tcpa:
        vit_extractor.train()
    total_loss = 0.0
    class_loss_sum = 0.0
    concept_loss_sum = 0.0
    orth_loss_sum = 0.0
    reg_loss_sum = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        class_labels = batch['class_label'].to(device)
        concept_labels = batch['concept_labels'].to(device)
        batch_size = images.size(0)
        if tune_tcpa:
            patch_features = vit_extractor(images)
            concept_features = clip_encoder(batch_size)
        else:
            with torch.no_grad():
                patch_features = vit_extractor(images)
                concept_features = clip_encoder(batch_size)
        visual_prior = None
        concept_prior = None
        class_logits, concept_activation, orth_loss = model(
            patch_features, concept_features,
            visual_prior, concept_prior
        )
        conc_extra = torch.tensor(0.0, device=device)
        equiv_extra = torch.tensor(0.0, device=device)
        center_extra = torch.tensor(0.0, device=device)
        if use_beyond_losses and getattr(model, "use_parts", False):
            z_parts, w_parts, s_fg, maps = model.get_parts(patch_features)
            B = images.size(0)
            if maps.shape[-1] > 1:
                from lib import landmark_coordinates, batch_rigid_transform
                from losses import conc_loss
                maps_parts = maps[:, :-1, :, :]
                loc_x, loc_y, grid_x, grid_y = landmark_coordinates(maps_parts, device)
                conc_extra = conc_loss(loc_x, loc_y, grid_x, grid_y, maps_parts) * l_conc
                angle = torch.rand(B).to(device) * 180 - 90
                translate = torch.rand(B, 2).to(device) * 0.2 - 0.1
                scale = torch.rand(B).to(device) * 0.5 + 0.9
                transf_img, _ = batch_rigid_transform(images, angle.cpu().numpy(), translate.cpu().numpy(), scale=scale.cpu().numpy(), invert=False, device=device)
                if tune_tcpa:
                    patch_features_t = vit_extractor(transf_img.to(device))
                else:
                    with torch.no_grad():
                        patch_features_t = vit_extractor(transf_img.to(device))
                _, _, _, maps_t = model.get_parts(patch_features_t)
                rot_back, maskI = batch_rigid_transform(maps_t, angle.cpu().numpy(), translate.cpu().numpy(), scale=scale.cpu().numpy(), invert=True, device=device)
                maskI = maskI[:, :1]
                import torch.nn.functional as Fnn
                cos_sim = Fnn.cosine_similarity((maps_parts*maskI) .reshape(B, maps_parts.shape[1], -1),
                                                (rot_back[:, :-1, :, :]*maskI).reshape(B, maps_parts.shape[1], -1), dim=-1)
                equiv_extra = (1.0 - cos_sim.mean()) * l_equiv
            if criterion_center is not None and w_parts.shape[1] > 0:
                center_extra = criterion_center(z_parts, class_labels) * l_center
        class_logits_detached = model.concept_to_class(concept_activation.detach())
        loss_class = criterion_class(class_logits_detached, class_labels)
        loss_concept = criterion_concept(concept_activation, concept_labels)
        reg_dict = model.concept_to_class.reg_loss() if hasattr(model.concept_to_class, "reg_loss") else {}
        reg_sum = sum(reg_dict.values()) if len(reg_dict) > 0 else torch.tensor(0.0, device=device)
        loss = loss_class + lambda_concept * loss_concept + lambda_orth * orth_loss + lambda_reg * reg_sum + conc_extra + equiv_extra + center_extra
        optimizer.zero_grad()
        loss.backward()
        if tune_tcpa and hasattr(vit_extractor, "tcpas"):
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(vit_extractor.tcpas.parameters()),
                max_norm=1.0
            )
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        class_loss_sum += loss_class.item()
        concept_loss_sum += loss_concept.item()
        orth_loss_sum += orth_loss.item()
        reg_loss_sum += (reg_sum.item() if isinstance(reg_sum, torch.Tensor) else float(reg_sum))
        _, predicted = torch.max(class_logits, 1)
        total += class_labels.size(0)
        correct += (predicted == class_labels).sum().item()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    metrics = {
        'loss': total_loss / len(train_loader),
        'class_loss': class_loss_sum / len(train_loader),
        'concept_loss': concept_loss_sum / len(train_loader),
        'orth_loss': orth_loss_sum / len(train_loader),
        'reg_loss': reg_loss_sum / len(train_loader),
        'accuracy': 100.0 * correct / total
    }
    return metrics


@torch.no_grad()
def evaluate(
    model: DOT_CBM,
    vit_extractor: ViT_PatchExtractor,
    clip_encoder: CLIP_ConceptEncoder,
    data_loader,
    criterion_class: nn.Module,
    criterion_concept: nn.Module,
    criterion_center: Optional[nn.Module],
    device: torch.device,
    lambda_orth: float = 0.1,
    lambda_concept: float = 1.0,
    lambda_reg: float = 1.0
) -> dict:
    model.eval()
    vit_extractor.eval()
    total_loss = 0.0
    class_loss_sum = 0.0
    concept_loss_sum = 0.0
    reg_loss_sum = 0.0
    correct = 0
    total = 0
    concept_preds_list = []
    concept_labels_list = []
    pbar = tqdm(data_loader, desc='Evaluating')
    for batch in pbar:
        images = batch['image'].to(device)
        class_labels = batch['class_label'].to(device)
        concept_labels = batch['concept_labels'].to(device)
        batch_size = images.size(0)
        patch_features = vit_extractor(images)
        concept_features = clip_encoder(batch_size)
        class_logits, concept_activation, orth_loss = model(
            patch_features, concept_features
        )
        class_labels = class_labels.to(class_logits.device, non_blocking=True).long()
        loss_class = criterion_class(class_logits.float(), class_labels)
        concept_labels = concept_labels.to(concept_activation.device, non_blocking=True).to(dtype=concept_activation.dtype)
        loss_concept = criterion_concept(concept_activation, concept_labels)
        reg_dict = model.concept_to_class.reg_loss() if hasattr(model.concept_to_class, "reg_loss") else {}
        reg_sum = sum(reg_dict.values()) if len(reg_dict) > 0 else torch.tensor(0.0, device=device)
        loss = loss_class + lambda_concept * loss_concept + lambda_orth * orth_loss + lambda_reg * reg_sum
        total_loss += loss.item()
        class_loss_sum += loss_class.item()
        concept_loss_sum += loss_concept.item()
        reg_loss_sum += (reg_sum.item() if isinstance(reg_sum, torch.Tensor) else float(reg_sum))
        _, predicted = torch.max(class_logits, 1)
        total += class_labels.size(0)
        correct += (predicted == class_labels).sum().item()
        concept_preds = (concept_activation >= 0.5).float()
        concept_preds_list.append(concept_preds.cpu())
        concept_labels_list.append(concept_labels.cpu())
    all_concept_preds = torch.cat(concept_preds_list, dim=0)
    all_concept_labels = torch.cat(concept_labels_list, dim=0)
    concept_accuracy = (all_concept_preds == all_concept_labels).float().mean().item() * 100
    pred_np = all_concept_preds.numpy().astype(int)
    target_np = all_concept_labels.numpy().astype(int)
    overall_match = sklearn.metrics.accuracy_score(target_np, pred_np) * 100
    f1_list = []
    for i in range(target_np.shape[-1]):
        true_vars = target_np[:, i]
        pred_vars = pred_np[:, i]
        f1 = sklearn.metrics.f1_score(true_vars, pred_vars, average='macro')
        f1_list.append(f1)
    concept_f1_macro = float(np.mean(f1_list)) * 100
    metrics = {
        'loss': total_loss / len(data_loader),
        'class_loss': class_loss_sum / len(data_loader),
        'concept_loss': concept_loss_sum / len(data_loader),
        'reg_loss': reg_loss_sum / len(data_loader),
        'accuracy': 100.0 * correct / total,
        'concept_accuracy': concept_accuracy,
        'concept_overall_accuracy': overall_match,
        'concept_f1_macro': concept_f1_macro
    }
    return metrics


def load_concept_names_from_predicates(data_dir):
    names = []
    with open(os.path.join(data_dir, "predicates.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                names.append(parts[1].strip())
    return names


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'dotcbm_awa2_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    config = vars(args)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    log_path = os.path.join(output_dir, 'train_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('epoch,train_loss,train_acc,train_class_loss,train_concept_loss,train_orth_loss,train_reg_loss,val_loss,val_acc,val_concept_acc,val_concept_overall_acc,val_concept_f1,val_reg_loss,lr\n')
    args.dataset = "awa2"
    train_loader, val_loader, test_loader = get_dataset(args)
    sample_batch = next(iter(train_loader))
    num_concepts = sample_batch['concept_labels'].shape[1]
    num_classes = len(train_loader.dataset.class_to_index)
    print(f"训练集大小: {len(train_loader.dataset)} | 验证集大小: {len(val_loader.dataset)} | 测试集大小: {len(test_loader.dataset)}")
    print(f"类别数: {num_classes} | 概念数: {num_concepts}")
    concept_list = load_concept_names_from_predicates(args.data_dir)
    clip_encoder = CLIP_ConceptEncoder(
        concept_list=concept_list,
        model_name=args.clip_model,
        device=device,
        pretrained_path=args.clip_pretrained
    )
    if args.tune_tcpa:
        vit_extractor = ViT_PatchExtractor_TCPA(
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
        vit_extractor = ViT_PatchExtractor(
            model_name=args.vit_model,
            weights_path=(getattr(args, 'vit_weights_path', None) or None),
            pretrained=False,
            freeze=True
        ).to(device)
    model = DOT_CBM(
        num_patches=196,
        num_concepts=num_concepts,
        num_classes=num_classes,
        patch_dim=vit_extractor.embed_dim,
        concept_dim=clip_encoder.embed_dim,
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
    criterion_class = nn.CrossEntropyLoss()
    criterion_concept = nn.BCELoss()
    criterion_center = None
    if args.use_parts and device.type == 'cuda':
        try:
            from losses import PartCenterLoss
            criterion_center = PartCenterLoss(num_classes=num_classes, num_parts=args.num_parts, feat_dim=args.hidden_dim, use_gpu=(device.type=='cuda'))
        except Exception:
            criterion_center = None
    if args.tune_tcpa:
        optimizer = optim.Adam(
            [
                {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
                {"params": vit_extractor.tcpas.parameters(), "lr": args.lr_tcpa, "weight_decay": args.weight_decay},
            ]
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    best_val_class_acc = 0.0
    best_val_concept_acc = 0.0
    patience_counter = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_metrics = train_epoch(
            model, vit_extractor, clip_encoder,
            train_loader, optimizer,
            criterion_class, criterion_concept, criterion_center,
            device, args.lambda_orth, args.lambda_concept, args.lambda_reg,
            use_priors=args.use_priors,
            tune_tcpa=args.tune_tcpa,
            use_beyond_losses=True,
            l_conc=args.l_conc,
            l_equiv=args.l_equiv,
            l_center=args.l_center
        )
        val_metrics = evaluate(
            model, vit_extractor, clip_encoder,
            val_loader,
            criterion_class, criterion_concept, criterion_center,
            device, args.lambda_orth, args.lambda_concept, args.lambda_reg
        )
        scheduler.step()
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch + 1},{train_metrics['loss']:.6f},{train_metrics['accuracy']:.2f},{train_metrics['class_loss']:.6f},{train_metrics['concept_loss']:.6f},{train_metrics['orth_loss']:.6f},{train_metrics['reg_loss']:.6f},{val_metrics['loss']:.6f},{val_metrics['accuracy']:.2f},{val_metrics['concept_accuracy']:.2f},{val_metrics['concept_overall_accuracy']:.2f},{val_metrics['concept_f1_macro']:.2f},{val_metrics['reg_loss']:.6f},{optimizer.param_groups[0]['lr']:.8f}\n")
        print(f"Val   - Top1 Acc: {val_metrics['accuracy']:.2f}%, "
              f"Concept Acc: {val_metrics['concept_accuracy']:.2f}%, "
              f"Concept Overall Acc: {val_metrics['concept_overall_accuracy']:.2f}%, "
              f"Concept F1(macro): {val_metrics['concept_f1_macro']:.2f}%")
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('val/concept_accuracy', val_metrics['concept_accuracy'], epoch)
        writer.add_scalar('val/concept_overall_accuracy', val_metrics['concept_overall_accuracy'], epoch)
        writer.add_scalar('val/concept_f1_macro', val_metrics['concept_f1_macro'], epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        if val_metrics['accuracy'] > best_val_class_acc:
            best_val_class_acc = val_metrics['accuracy']
            payload = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_class_acc,
            }
            if hasattr(vit_extractor, 'tcpas'):
                payload['tcpa_state_dict'] = vit_extractor.tcpas.state_dict()
                torch.save(vit_extractor.tcpas.state_dict(), os.path.join(output_dir, 'best_tcpa_class.pth'))
            torch.save(payload, os.path.join(output_dir, 'best_class_model.pth'))
            print(f"Saved best class model at epoch {epoch + 1}: Top1 {best_val_class_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        if val_metrics['concept_accuracy'] > best_val_concept_acc:
            best_val_concept_acc = val_metrics['concept_accuracy']
            payload_c = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_concept_accuracy': best_val_concept_acc,
            }
            if hasattr(vit_extractor, 'tcpas'):
                payload_c['tcpa_state_dict'] = vit_extractor.tcpas.state_dict()
                torch.save(vit_extractor.tcpas.state_dict(), os.path.join(output_dir, 'best_tcpa_concept.pth'))
            torch.save(payload_c, os.path.join(output_dir, 'best_concept_model.pth'))
            print(f"Saved best concept model at epoch {epoch + 1}: Concept Acc {best_val_concept_acc:.2f}%")
        if args.patience > 0 and patience_counter >= args.patience:
            break
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pth'))
    checkpoint = torch.load(os.path.join(output_dir, 'best_class_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics_mixed = None
    test_metrics_file = None
    if hasattr(vit_extractor, 'tcpas'):
        if 'tcpa_state_dict' in checkpoint:
            vit_extractor.tcpas.load_state_dict(checkpoint['tcpa_state_dict'])
            test_metrics_mixed = evaluate(
                model, vit_extractor, clip_encoder,
                test_loader,
                criterion_class, criterion_concept, criterion_center,
                device, args.lambda_orth, args.lambda_concept, args.lambda_reg
            )
        tcpa_path = os.path.join(output_dir, 'best_tcpa_class.pth')
        if os.path.exists(tcpa_path):
            vit_extractor.tcpas.load_state_dict(torch.load(tcpa_path))
            test_metrics_file = evaluate(
                model, vit_extractor, clip_encoder,
                test_loader,
                criterion_class, criterion_concept, criterion_center,
                device, args.lambda_orth, args.lambda_concept, args.lambda_reg
            )
    final_metrics = test_metrics_file or test_metrics_mixed
    if final_metrics is None:
        final_metrics = evaluate(
            model, vit_extractor, clip_encoder,
            test_loader,
            criterion_class, criterion_concept, criterion_center,
            device, args.lambda_orth, args.lambda_concept, args.lambda_reg
        )
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"TEST,{final_metrics['loss']:.6f},{final_metrics['accuracy']:.2f},,,,{final_metrics['loss']:.6f},{final_metrics['accuracy']:.2f},{final_metrics['concept_accuracy']:.2f},{final_metrics.get('concept_overall_accuracy', 0.0):.2f},{final_metrics['concept_f1_macro']:.2f},\n")
    writer.close()
    print(f"Test  - Top1 Acc: {final_metrics['accuracy']:.2f}%, "
          f"Concept Acc: {final_metrics['concept_accuracy']:.2f}%, "
          f"Concept Overall Acc: {final_metrics.get('concept_overall_accuracy', 0.0):.2f}%, "
          f"Concept F1(macro): {final_metrics['concept_f1_macro']:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DOT-CBM on AwA2 (data_util)')
    parser.add_argument('--data_dir', type=str, required=False, default='D:/BaiduNetdiskDownload/OTCBM-main/AwA2-data/Animals_with_Attributes2')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--vit_model', type=str, default='vit_base_patch16_224')
    parser.add_argument('--clip_model', type=str, default='ViT-B-32')
    parser.add_argument('--clip_pretrained', type=str, required=False, default='D:/BaiduNetdiskDownload/OTCBM-main/open_clip_pytorch_model.bin')
    parser.add_argument('--vit_weights_path', type=str, required=False, default='D:\\BaiduNetdiskDownload\\OTCBM-main\\vit_base_p16_224-80ecf9dd.pth')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--ot_reg', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation_method', type=str, default='weighted')
    parser.add_argument('--activation_tau', type=float, default=1)
    parser.add_argument('--use_parts', action='store_true', default=False)
    parser.add_argument('--num_parts', type=int, default=12)
    parser.add_argument('--part_tv_weight', type=float, default=0.1)
    parser.add_argument('--part_distinct_weight', type=float, default=0.1)
    parser.add_argument('--part_presence_weight', type=float, default=0.1)
    parser.add_argument('--l_conc', type=float, default=0.1)
    parser.add_argument('--l_equiv', type=float, default=0.1)
    parser.add_argument('--l_center', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tune_tcpa', action='store_true', default=True)
    parser.add_argument('--lr_tcpa', type=float, default=1e-4)
    parser.add_argument('--tcpa_last_k', type=int, default=12)
    parser.add_argument('--tcpa_layers', type=int, nargs='*')
    parser.add_argument('--tcpa_n_img_prompts', type=int, default=20)
    parser.add_argument('--tcpa_n_cls_prompts', type=int, default=4)
    parser.add_argument('--tcpa_topk', type=int, default=4)
    parser.add_argument('--tcpa_tau', type=float, default=0.1)
    parser.add_argument('--tcpa_dropout', type=float, default=0.1)
    parser.add_argument('--tcpa_no_mlp', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lambda_orth', type=float, default=0.1)
    parser.add_argument('--lambda_concept', type=float, default=5.0)
    parser.add_argument('--lambda_reg', type=float, default=1.0)
    parser.add_argument('--use_priors', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--concept_head', type=str, choices=['nam', 'mlp'], default='nam')
    args = parser.parse_args()
    train(args)

