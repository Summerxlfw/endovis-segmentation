# curated_modules 子集说明
> 用于双路径（Offline/Online）息肉分割实验的精简模块集合，便于在大仓之外快速查阅/打包。依赖未裁剪，落地前需用 dummy forward 验证导入、形状与依赖。

## 已收录（四份文档提到的候选全集）
- Backbone（ Teacher/Student 候选）：UniRepLKNet、TransNext、rmt、pkinet、lsknet、inceptionnext、hgnetv2 全系（含 _star/_DRC/_LoGStem/_LWGA/_MANet_iRMB）、fasternet、starnet、EfficientFormerV2、mobilenetv4、VanillaNet、revcol、dinov3、dinov3_adapter
- Neck/FPN/解码相关：afpn/YOLO11（AFPN_P2345/AFPN_P345）、GoldYOLO、HyperACE、FDPN、HS_FPN
- Module：efficientvim（EfficientViMBlock/ConvLayer1D）、hcfnet（DASI/PPA）、kernel_warehouse（KWConv1d/3d）、kat_1dgroup（KAT_Group）、FreqFusion、EfficientFormerV2、metaformer、mhafyolo（RepHMS）、cfpt
- Transformer：SHSA、LRSA、TAB、GSA、biformer、SwinTransformer、transformer（MSDeformAttn 相关）、rtdetrv2_decoder、dfine_decoder、dqs_dfine_decoder
- SSM_Mamba：mamba2、mamba_simple、SS2D、VSSD、mambaout

## 使用提醒
- 路径与导入未实测：若要运行，请先在项目根目录做 dummy forward（假特征输入），确认 1/4–1/32 输出形状、通道与依赖。
- 自定义算子/依赖：Mamba/Triton、MSDeformable、kat_rational_cu 等可能需编译；如不可用可改为纯 PyTorch 组合（EfficientViMBlock + SHSA/LRSA + AFPN）。
- 分割头：大部分为 Neck，需要在 1/4 或 1/8 补上采样 + 3×3/1×1 分类头。
- 蒸馏对齐：建议保持 1/4、1/8、1/16 做 feature KD，1/32 做 logit KD；时序块挂在 1/8/1/16。
