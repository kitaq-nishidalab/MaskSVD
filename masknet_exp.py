import torch
import torch.onnx
from network import MaskNet  # MaskNetのモデル定義をインポート

# モデル全体をロード（保存されたモデルインスタンスをそのままロード）
masknet_model = torch.load("checkpoint/model_weight_epoch300_batchsize32_plane.pth")
masknet_model.eval()  # 推論モードに切り替える


# ダミー入力を作成
N = 1024  # 点群の点数（例として1024を設定）
template = torch.randn(1, N, 3).cuda()  # ダミーのテンプレート点群
source = torch.randn(1, N, 3).cuda()    # ダミーのソース点群


# モデルのONNXエクスポート
torch.onnx.export(
    masknet_model,                        # モデル
    (template, source),                   # 2つの点群をタプルで渡す
    "onnx/masknet.onnx",                   # 出力ファイル名
    export_params=True,                   # モデルのパラメータをONNXファイルに含める
    opset_version=11,                     # opsetバージョン
    do_constant_folding=True,             # 定数フォールディングを有効にして最適化
    input_names=['template', 'source'],   # 入力名
    output_names=['masked_template', 'predicted_mask'],  # 出力名
    dynamic_axes={
        'template': {1: 'num_points'},    # 点数が動的に変わる可能性に対応
        'source': {1: 'num_points'},      # 点数が動的に変わる可能性に対応
        'masked_template': {1: 'num_points'},     # バッチサイズの動的対応
        'predicted_mask': {1: 'num_points'} 
    }
)
