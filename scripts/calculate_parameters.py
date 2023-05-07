def calculate_mt5_params(encoder_layers, decoder_layers, vocab_size, embed_dim, ffn_dim, model_name="MNMT"):
    embed_params = embed_dim * vocab_size
    encoder_params = (embed_dim * embed_dim * 4 + ffn_dim * embed_dim * 2 + embed_dim * 2) * encoder_layers
    decoder_params = (embed_dim * embed_dim * 8 + ffn_dim * embed_dim * 2 + embed_dim * 2) * decoder_layers
    params = embed_params + encoder_params + decoder_params
    print(f"w/ embed: {model_name}: {params // 1000000}")
    params = encoder_params + decoder_params
    print(f"w/o embed: {model_name}: {params // 1000000}")


def calculate_standard_transformer_params(encoder_layers, decoder_layers, vocab_size, embed_dim, ffn_dim,
                                          model_name="MNMT"):
    embed_params = embed_dim * vocab_size
    encoder_params = (embed_dim * embed_dim * 4 + ffn_dim * embed_dim * 2 + embed_dim * 2) * encoder_layers
    decoder_params = (embed_dim * embed_dim * 8 + ffn_dim * embed_dim * 2 + embed_dim * 2) * decoder_layers
    params = embed_params + encoder_params + decoder_params
    print(f"w/ embed: {model_name}: {params // 1000000}")
    params = encoder_params + decoder_params
    print(f"w/o embed: {model_name}: {params // 1000000}")


def calculate_interleaved_transformer_params(encoder_layers, decoder_layers, vocab_size, embed_dim, model_name="MNMT"):
    embed_params = embed_dim * vocab_size
    encoder_params = (embed_dim * embed_dim * 4 + embed_dim * 2 + embed_dim * 4 * embed_dim * 2) * encoder_layers
    decoder_params = (
                                 embed_dim * embed_dim * 4 * 2 + embed_dim * 2 + embed_dim * 4 * embed_dim * 2 * 2) * decoder_layers
    params = embed_params + encoder_params + decoder_params
    print("{}: {}".format(model_name, params))


def calculate_our_params(encoder_layers, decoder_layers, vocab_size, embed_dim, ls_number=4, model_name="MNMT"):
    embed_params = embed_dim * vocab_size
    encoder_params = (embed_dim * embed_dim * 4 + embed_dim * 2 + embed_dim * 4 * embed_dim * 2) * encoder_layers
    decoder_params = (
                                 embed_dim * embed_dim * 4 * 2 + embed_dim * 2 + embed_dim * 4 * embed_dim * 2 * 2) * decoder_layers
    ls_params = (embed_dim + embed_dim * 4 * embed_dim * 2) * ls_number
    params = embed_params + encoder_params + decoder_params + ls_params
    print("{}: {}".format(model_name, params))


def calculate_monolingual_adapter_params(encoder_layers, decoder_layers, vocab_size, embed_dim, ls_number=11,
                                         ls_dim=256, model_name="monolingual_adapter"):
    embed_params = embed_dim * vocab_size
    encoder_params = (embed_dim * embed_dim * 4 + embed_dim * 2 + embed_dim * 4 * embed_dim * 2) * encoder_layers
    decoder_params = (
                                 embed_dim * embed_dim * 4 * 2 + embed_dim * 2 + embed_dim * 4 * embed_dim * 2 * 2) * decoder_layers
    ls_params = (embed_dim + embed_dim * ls_dim * 2) * ls_number * 2
    params = embed_params + encoder_params + decoder_params + ls_params
    print("{}: {}".format(model_name, params))


# xlmr base
calculate_mt5_params(encoder_layers=12, decoder_layers=0, vocab_size=250000, embed_dim=768, ffn_dim=768 * 2,
                     model_name="xlmr base")

# mT5 base
calculate_mt5_params(encoder_layers=12, decoder_layers=12, vocab_size=250000, embed_dim=768, ffn_dim=768 * 2,
                     model_name="mT5 base")
# mT5 large
calculate_mt5_params(encoder_layers=12, decoder_layers=12, vocab_size=250000, embed_dim=768, ffn_dim=768 * 2,
                     model_name="mT5 base")

calculate_standard_transformer_params(encoder_layers=12, decoder_layers=12, vocab_size=64000, embed_dim=768,
                                      ffn_dim=768 * 4, model_name="base")

# v6
calculate_standard_transformer_params(encoder_layers=12, decoder_layers=12, vocab_size=250000, embed_dim=768,
                                      ffn_dim=768 * 4, model_name="large")
calculate_standard_transformer_params(encoder_layers=12, decoder_layers=24, vocab_size=250000, embed_dim=768,
                                      ffn_dim=768 * 4, model_name="large")
calculate_standard_transformer_params(encoder_layers=12, decoder_layers=16, vocab_size=250000, embed_dim=768,
                                      ffn_dim=768 * 4, model_name="large")
calculate_standard_transformer_params(encoder_layers=12, decoder_layers=28, vocab_size=250000, embed_dim=768,
                                      ffn_dim=768 * 4, model_name="large")
