# Krea2 Slider prompts

Krea2公式の [`docs/prompting.md`](../参考/krea-2/docs/prompting.md) に従い、人物・画風・衣装・構図・背景・光を記述する英語の自然文にしています。アニメ版はアニメ画風と、概念以外の体型・衣服の変化を抑えるMageFlow改訂版の記述を引き継ぎました。aging/deagingには写真調の肌・布の質感を使う実写風版もあります。

| ファイル | 正のLoRA強度の方向 | 参照元 |
|---|---|---|
| [anime_aging_slider_fullbody.yaml](anime_aging_slider_fullbody.yaml) | アニメ調で成人女性の顔・首・手の皮膚を加齢させる | MageFlow aging v2 |
| [realistic_aging_slider_fullbody.yaml](realistic_aging_slider_fullbody.yaml) | 実写風で成人女性の顔・首・手の皮膚を加齢させる | anime agingの写真調版 |
| [breast_size_slider.yaml](breast_size_slider.yaml) | 着衣の成人女性の胸部ボリュームを中程度から大きめへ | MageFlow breast v5 |
| [breast_size_slider_v2.yaml](breast_size_slider_v2.yaml) | 胸部ボリュームの大小を服の輪郭で明示する改訂版 | Krea2 breast size改訂版 |
| [clothing_fit_tighten_slider.yaml](clothing_fit_tighten_slider.yaml) | 標準的な服から、体に沿う仕立て・滑らかな輪郭へ（既存版） | Anima clothing fit v2 |
| [clothing_fit_loosen_slider.yaml](clothing_fit_loosen_slider.yaml) | 標準的な服から、ゆったりした仕立て・余り布の折り目へ（既存版） | Anima clothing fit v3 |
| [clothing_fit_tighten_slider_v2.yaml](clothing_fit_tighten_slider_v2.yaml) | ゆったりした服との対比で、布と身体の隙間を小さくする（改訂版） | Krea2 clothing fit改訂版 |
| [clothing_fit_loosen_slider_v2.yaml](clothing_fit_loosen_slider_v2.yaml) | 密着した服との対比で、布と身体の隙間を大きくする（改訂版） | Krea2 clothing fit改訂版 |
| [anime_deaging_slider_fullbody.yaml](anime_deaging_slider_fullbody.yaml) | アニメ調で着衣の成人女性から幼児へ、年齢・顔・頭身・手足の比率を変える | MageFlow deaging v2 |
| [realistic_deaging_slider_fullbody.yaml](realistic_deaging_slider_fullbody.yaml) | 実写風で着衣の成人女性から幼児へ、年齢・顔・頭身・手足の比率を変える | anime deagingの写真調版 |
| [female_size_increase_slider.yaml](female_size_increase_slider.yaml) | 通常サイズの成人男性との構図で、成人女性の全身を比例拡大する | Krea2用の新規プロンプト設計 |
| [female_size_decrease_slider.yaml](female_size_decrease_slider.yaml) | 通常サイズの成人男性との構図で、成人女性の全身を比例縮小する | Krea2用の新規プロンプト設計 |

各ファイルは**学習用6件**です。aging/deagingは全身4件＋腰上2件、breast初版は全身4件＋膝下まで2件、breast v2は頭から腰下まで4件＋全身2件です。clothing fitの既存版はAnimaの学習用0〜5を引き継ぎ、上半身2件＋腰上2件＋腿上2件です。clothing fit改訂版は同じ6種類の衣装・構図をもとに、頭から腰下まで5件＋膝下まで1件へ広げています。参照元の評価用6/7は収録していません。現在のKrea2ノードは渡された全件を学習に使用し、学習・評価indicesの指定を持たないためです。

Female sizeの2セットも各6件で、すべて成人男女2人の全身構図です。女性が左の3件と右の3件、正面3件と斜め3件、3種類の衣装を組み合わせています。

## 読み込み方

`Krea2 Slider Train LoRA`の **prompt_yaml** 一覧からファイル名を選択します。ネイティブのMODEL／CLIPも同じ学習ノードへ接続します。テキストの貼り付けは不要です。

ファイルは通常のYAMLブロック形式で保存しています。`>-`による折り返しとYAMLアンカーも読み込めます。今回の形式変更でプロンプト本文は変えていません。JSON互換のYAMLも引き続き読み込めます。追加ファイルはこのフォルダー内の`.yaml`／`.yml`が一覧の対象です。

フィールドは現行ノードに合わせた`target`、`positive`、`negative`、`neutral`だけです。MageFlow／Animaの`unconditional`をKrea2の`negative`へ対応づけています。移植した既存版は`target = negative = neutral`で、`positive`だけに学習したい差分があります。breast v2、clothing fit改訂版、female sizeは`target = neutral`を基準に、`positive`と`negative`を対極の概念にしています。breast v2は中程度の胸部ボリュームを基準に、大きめ／小さめを対比します。`negative`は比較対象の概念であり、生成時の除外語句リストではありません。

Krea2の [`encoder.py`](../参考/krea-2/encoder.py) はsystem/userテンプレートを内部で付加します。ファイル内にはチャット用の特殊トークン、system指示、品質スコアタグを入れていません。

## ノード側の設定

MageFlow／Animaの`guidance_scale`、`action`、解像度、`batch_size`はKrea2のプロンプトレコードには入れません。

各セットは **training_direction=single** を開始設定にします。参照元の`action=enhance`に合わせて、positiveへ向かう方向を強度`+1`で1回だけ学習します。`negative`は比較基準であり、逆方向の学生学習を意味しません。agingの比較基準は成人の肌で、幼児プロンプトは使いません。

| セット | eta | teacher_guidance_scale | teacher_norm_reference |
|---|---:|---:|---|
| aging | 1 | 1 | none |
| breast size | 1 | 2 | none |
| deaging | 1 | 1 | none |

この係数は参照元の概念差分の強さを対応づけた開始値であり、Krea2での最適値や同じ画質を保証するものではありません。胸サイズの効果が強すぎる場合は`teacher_guidance_scale=1`も比較対象になります。

16GB向けの動作確認はノードの既定値である512×512、rank/alpha 8/8、INT8、checkpoint ON、CPU退避16ブロックから開始できます。参照元の1024×1024／896×1152をファイルから自動適用する処理はありません。

## Aging/deagingの画風別ファイル

既存の`aging_slider_fullbody.yaml`を`anime_aging_slider_fullbody.yaml`へ、`deaging_slider_fullbody.yaml`を`anime_deaging_slider_fullbody.yaml`へ改名しました。アニメ版のプロンプト本文はそのままです。保存済みワークフローで旧名を使っている場合は、`prompt_yaml`で新しい名前を選び直してください。

追加した`realistic_aging_slider_fullbody.yaml`／`realistic_deaging_slider_fullbody.yaml`は、同じ6件の年齢差分・衣装・姿勢・構図・背景・照明を引き継ぎ、画風を実写風のスタジオ写真へ変更しています。全フィールドで写真、自然な肌の質感、現実的な布の質感を共通にし、deagingの`matte skin shading`も肌の質感の表現へ置き換えています。

agingは成人の顔・首・手の皮膚の加齢、deagingは成人から幼児への顔・頭身・手足の比率の変化を学習する構成です。deagingは成人の範囲内だけの若返りではありません。両者とも着衣の構図を維持し、`target = negative = neutral`を基準に、`positive`へ向かう1方向の学習にします。

`training_direction=single`で画風・方向ごとに別々に学習し、`output_name`も`krea2_realistic_aging`／`krea2_realistic_deaging`などに分けてください。実写風版の学習・画像生成は未実施で、肌や衣装の保持とSlider効果は画像で未評価です。

## Clothing fit tighten/loosen

`breast_size_slider_v2.yaml`の自然文・画風・背景の書き方を参考に、Animaのタグ列を英語の文章へ変換しています。胸の大小の対比は移さず、各レコードでは服のフィット感を説明する2文だけを変更しています。人物の体格、髪、衣装の種類・色、姿勢、構図、背景、照明はペア内で同一です。

元の学習用6件にあるブラウス、カーディガン、ジャケット＋スカート、セーター、長袖トップス＋スカート、カジュアルドレスと、正面／斜めの構図を維持しています。既存の`clothing_fit_slider_v2.yaml`を`clothing_fit_tighten_slider.yaml`へ、`clothing_fit_slider_v3.yaml`を`clothing_fit_loosen_slider.yaml`へ改名しました。既存版のプロンプト本文はそのままで、tightenは体に沿う仕立て・滑らかな布の輪郭、loosenはゆったりした仕立て・余り布の折り目・箱形の輪郭へ進む独立したLoRA用です。

追加した`clothing_fit_tighten_slider_v2.yaml`と`clothing_fit_loosen_slider_v2.yaml`は、`target = neutral`を通常のフィット感にし、密着／ゆったりを対比する改訂版です。tightenは`positive=密着`、`negative=ゆったり`、loosenはその逆です。通常状態の説明には対極の語を混ぜず、布の余裕と袖幅を記述しています。

改訂版は`tailored`、`boxy`、折り目の量に頼る表現を外し、布と身体の隙間、腰に沿う／離れる輪郭、腕に沿う／広い袖の違いに絞っています。カーディガン・ジャケットは内側のトップスの上でのフィット感を記述しています。トップス類は腰下まで、膝丈のドレスは膝下までを映し、袖・腰・衣服の輪郭が見える構図です。各レコードのフィールド間で異なるのはフィット感の2文だけです。

改訂版の`_v2`は各方向のKrea2プロンプト改訂番号です。移植元Animaのv2/v3とは別です。保存済みのワークフローで旧ファイル名を選んでいる場合は、`prompt_yaml`で新しい名前を選び直してください。

`prompt_yaml`で選択し、`training_direction=single`で別々に学習します。どちらの方向も正のLoRA強度を使います。改訂版の`output_name`は`krea2_clothing_fit_tighten_v2`／`krea2_clothing_fit_loosen_v2`などに分けてください。学習係数や解像度はノード側で指定します。Animaの設定値は自動適用されません。

YAML・現行parserへの適合と差分の構造を検証対象にしています。服や体型の保持、学習後のSlider効果は画像で未評価です。

## Female size increase/decrease

`breast_size_slider_v2.yaml`のように、冒頭で女性の全身サイズを具体的に対比する強い記述へ改訂しています。`target = neutral`は通常サイズの成人男女です。increaseは`positive`が男性の約2倍の巨大な女性、`negative`が約半分の小型の女性で、decreaseはその対比を逆にしています。男性は全フィールドで通常サイズです。

女性の頭・肩・胴・腕・手・脚・足と全身の輪郭が、同じ比率で拡大／縮小することを明記しています。成人の顔と頭身を維持する文も全フィールドで共通です。`negative`は反対側のサイズの比較対象であり、通常サイズのbaselineや生成時の除外語句リストではありません。

各レコードでは、導入直後にある女性のサイズと相対身長を説明する2文だけがフィールド間で異なります。男性の通常サイズ、両者の衣装・姿勢・左右配置、カメラ、背景、照明は共通です。同じ奥行き・床面に二人を立たせ、全身と周囲の余白を描写しています。これは女性のスケール差を学習させるための新規設計で、Civitaiの元LoRAの学習レシピを再現したものではありません。

`prompt_yaml`で各ファイルを選び、**`training_direction=single`で別々に学習**してください。両方とも正のLoRA強度で目的の方向へ進みます。縮小版も正の強度を使います。

反対側のサイズを`negative`に入れても、`single`では正の方向だけを学習します。プロンプト改訂の比較では、まず学習設定を同じにして対比の変更を評価してください。

| ファイル | output_nameの例 | 正のLoRA強度で狙う変化 |
|---|---|---|
| female_size_increase_slider.yaml | `krea2_female_size_increase` | 女性だけを大きくする |
| female_size_decrease_slider.yaml | `krea2_female_size_decrease` | 女性だけを小さくする |

プロンプト内の身長比は対比を明確にするための目安で、生成画像の比率を固定する値ではありません。実際の学習・画像生成は未実施です。同じseed・構図でLoRA強度を変え、女性／男性の身長比に加えて、男性自身の画面内サイズと背景の変化も比較してください。

## 確認範囲と出典

YAMLとしての読み込み、現行Krea2 parserへの適合、baselineの一致、目的以外のペア差分を確認済みです。ローカルComfyUIの`Krea2Tokenizer`で、テンプレート込みの最大トークン数はagingが335、breastが349、deagingが360でした。すべて512以内です。検証結果は`test-results/krea2-prompt-validation.json`に記録しています。

今回の変更では学習・画像生成は行っておらず、体型・衣装の保持やSlider品質は未評価です。

参照元ディレクトリ：`C:/ComfyUI/custom_nodes/Comfyui-mageflow-slider-node/prompts/`

- `prompts-mageflow-aging_slider_fullbody_v2.yaml`
- `prompts-mageflow-breast_size_slider_v5.yaml`
- `prompts-mageflow-deaging_slider_fullbody_v2.yaml`

Clothing fitの参照元ディレクトリ：`../Comfyui-anima-slider-node/prompts/`（このリポジトリのルートからの相対パス）

- `prompts-anima-clothing_fit_slider_v2.yaml`
- `prompts-anima-clothing_fit_slider_v3.yaml`

Krea2の記述方針：`参考/krea-2/docs/prompting.md`。入力テンプレートと最大長の根拠：`参考/krea-2/encoder.py`。公式の長いプロンプト例はTurboでの生成例であり、今回のRAW Slider学習の画質実証とは区別しています。
