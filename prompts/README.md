# Krea2 Slider prompts

Krea2公式の [`docs/prompting.md`](../参考/krea-2/docs/prompting.md) に従い、人物・画風・衣装・構図・背景・光を記述する英語の自然文にしています。アニメ画風と、概念以外の体型・衣服の変化を抑えるMageFlow改訂版の記述を引き継ぎました。

| ファイル | 正のLoRA強度の方向 | 参照元 |
|---|---|---|
| [aging_slider_fullbody.yaml](aging_slider_fullbody.yaml) | 成人女性の顔・首・手の皮膚を加齢させる | MageFlow aging v2 |
| [breast_size_slider.yaml](breast_size_slider.yaml) | 着衣の成人女性の胸部ボリュームを中程度から大きめへ | MageFlow breast v5 |
| [deaging_slider_fullbody.yaml](deaging_slider_fullbody.yaml) | 着衣の成人女性から幼児へ、年齢・顔・頭身・手足の比率を変える | MageFlow deaging v2 |

各ファイルは参照元の**学習用0〜5の6件**です。aging/deagingは全身4件＋腰上2件、breastは全身4件＋膝下まで2件です。参照元の評価用6/7は収録していません。現在のKrea2ノードは渡された全件を学習に使用し、学習・評価indicesの指定を持たないためです。

## 読み込み方

`Krea2 Slider Train LoRA`の **prompt_yaml** 一覧からファイル名を選択します。ネイティブのMODEL／CLIPも同じ学習ノードへ接続します。テキストの貼り付けは不要です。

ファイルは通常のYAMLブロック形式で保存しています。`>-`による折り返しとYAMLアンカーも読み込めます。今回の形式変更でプロンプト本文は変えていません。JSON互換のYAMLも引き続き読み込めます。追加ファイルはこのフォルダー内の`.yaml`／`.yml`が一覧の対象です。

フィールドは現行ノードに合わせた`target`、`positive`、`negative`、`neutral`だけです。MageFlowの`unconditional`をKrea2の`negative`へ対応づけています。すべて`target = negative = neutral`で、`positive`だけに学習したい差分があります。`negative`は比較対象の概念であり、生成時の除外語句リストではありません。

Krea2の [`encoder.py`](../参考/krea-2/encoder.py) はsystem/userテンプレートを内部で付加します。ファイル内にはチャット用の特殊トークン、system指示、品質スコアタグを入れていません。

## ノード側の設定

MageFlowの`guidance_scale`、`action`、解像度、`batch_size`はKrea2のプロンプトレコードには入れません。

3セットとも **training_direction=single** を既定にします。参照元の`action=enhance`に合わせて、positiveへ向かう方向を強度`+1`で1回だけ学習します。`negative`は比較基準であり、逆方向の学生学習を意味しません。agingの比較基準は成人の肌で、幼児プロンプトは使いません。

| セット | eta | teacher_guidance_scale | teacher_norm_reference |
|---|---:|---:|---|
| aging | 1 | 1 | none |
| breast size | 1 | 2 | none |
| deaging | 1 | 1 | none |

この係数は参照元の概念差分の強さを対応づけた開始値であり、Krea2での最適値や同じ画質を保証するものではありません。胸サイズの効果が強すぎる場合は`teacher_guidance_scale=1`も比較対象になります。

16GB向けの動作確認はノードの既定値である512×512、rank/alpha 8/8、INT8、checkpoint ON、CPU退避16ブロックから開始できます。参照元の1024×1024／896×1152をファイルから自動適用する処理はありません。

## 確認範囲と出典

YAMLとしての読み込み、現行Krea2 parserへの適合、baselineの一致、目的以外のペア差分を確認済みです。ローカルComfyUIの`Krea2Tokenizer`で、テンプレート込みの最大トークン数はagingが335、breastが349、deagingが360でした。すべて512以内です。検証結果は`test-results/krea2-prompt-validation.json`に記録しています。

今回の変更では学習・画像生成は行っておらず、体型・衣装の保持やSlider品質は未評価です。

参照元ディレクトリ：`C:/ComfyUI/custom_nodes/Comfyui-mageflow-slider-node/prompts/`

- `prompts-mageflow-aging_slider_fullbody_v2.yaml`
- `prompts-mageflow-breast_size_slider_v5.yaml`
- `prompts-mageflow-deaging_slider_fullbody_v2.yaml`

Krea2の記述方針：`参考/krea-2/docs/prompting.md`。入力テンプレートと最大長の根拠：`参考/krea-2/encoder.py`。公式の長いプロンプト例はTurboでの生成例であり、今回のRAW Slider学習の画質実証とは区別しています。
