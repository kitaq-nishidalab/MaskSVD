import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(TRT_LOGGER)

# Builder設定を取得
builder_config = builder.create_builder_config()

# メモリプール制限を設定（1GBに設定）
builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

# ネットワークの作成
with builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    # ONNXモデルの読み込み
    onnx_file_path = "onnx/masknet.onnx"
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print('ERROR: ONNX parsing failed')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit(1)

    # 最適化プロファイルの作成（動的な入力サイズを設定）
    profile = builder.create_optimization_profile()

    # 動的軸に合わせて、`num_points` と `batch_size` を設定
    input_name_template = "template"
    input_name_source = "source"
    
    # 入力の形状を確認し、動的範囲を設定
    # `num_points` が動的なので、それに合わせてプロファイルを設定
    N = 1024  # ダミーの点群数
    profile.set_shape(input_name_template, min=(1, N, 3), opt=(1, N, 3), max=(1, N, 3))
    profile.set_shape(input_name_source, min=(1, N, 3), opt=(1, N, 3), max=(1, N, 3))



    # プロファイルをbuilder_configに追加
    builder_config.add_optimization_profile(profile)

    # シリアライズされたネットワークをビルド
    serialized_network = builder.build_serialized_network(network, builder_config)

    # シリアライズされたエンジンを保存
    if serialized_network:
        engine_file_path = "trt/masknet.trt"
        with open(engine_file_path, "wb") as f:
            f.write(serialized_network)
    else:
        print("Failed to serialize the network.")
