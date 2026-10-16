# Krea2 Slider prompts

Krea2公式の [`docs/prompting.md`](../参考/krea-2/docs/prompting.md) に従い、人物・画風・衣装・構図・背景・光を記述する英語の自然文にしています。アニメ版はアニメ画風と、概念以外の体型・衣服の変化を抑えるMageFlow改訂版の記述を引き継ぎました。aging/deagingには写真調の肌・布の質感を使う実写風版もあります。

| ファイル | 正のLoRA強度の方向 | 参照元 |
|---|---|---|
| [anime_aging_slider_fullbody.yaml](anime_aging_slider_fullbody.yaml) | アニメ調で成人女性の顔・首・手の皮膚を加齢させる | MageFlow aging v2 |
| [realistic_aging_slider_fullbody.yaml](realistic_aging_slider_fullbody.yaml) | 実写風で成人女性の顔・首・手の皮膚を加齢させる | anime agingの写真調版 |
| [realistic_female_only_aging_slider_v1.yaml](realistic_female_only_aging_slider_v1.yaml) | 男女ペアで女性の顔・首・手の皮膚だけの加齢を狙う | 生成画像35枚を評価した女性限定候補 |
| [realistic_female_only_aging_slider_v2.yaml](realistic_female_only_aging_slider_v2.yaml) | 女性の皮膚の加齢を、配色・左右を交差した12件で学習する | 学習後の2構図・1seedで加齢と服の色保持を確認。裾・ポーズ等の変化は残る |
| [realistic_male_only_aging_slider_v1.yaml](realistic_male_only_aging_slider_v1.yaml) | 男女ペアで男性の顔・首・手の皮膚だけの加齢を狙う | 女性限定候補を基にした未生成の男性版 |
| [realistic_male_only_aging_slider_v2.yaml](realistic_male_only_aging_slider_v2.yaml) | 男性の皮膚の加齢を、配色・左右を交差した12件で学習する | 女性aging v2の方法を展開した未学習候補 |
| [breast_size_slider.yaml](breast_size_slider.yaml) | 着衣の成人女性の胸部ボリュームを中程度から大きめへ | MageFlow breast v5 |
| [breast_size_slider_v2.yaml](breast_size_slider_v2.yaml) | 胸部ボリュームの大小を服の輪郭で明示する改訂版 | Krea2 breast size改訂版 |
| [clothing_fit_tighten_slider.yaml](clothing_fit_tighten_slider.yaml) | 標準的な服から、体に沿う仕立て・滑らかな輪郭へ（既存版） | Anima clothing fit v2 |
| [clothing_fit_loosen_slider.yaml](clothing_fit_loosen_slider.yaml) | 標準的な服から、ゆったりした仕立て・余り布の折り目へ（既存版） | Anima clothing fit v3 |
| [clothing_fit_tighten_slider_v2.yaml](clothing_fit_tighten_slider_v2.yaml) | ゆったりした服との対比で、布と身体の隙間を小さくする（改訂版） | Krea2 clothing fit改訂版 |
| [clothing_fit_loosen_slider_v2.yaml](clothing_fit_loosen_slider_v2.yaml) | 密着した服との対比で、布と身体の隙間を大きくする（改訂版） | Krea2 clothing fit改訂版 |
| [anime_deaging_slider_fullbody.yaml](anime_deaging_slider_fullbody.yaml) | アニメ調で着衣の成人女性から幼児へ、年齢・顔・頭身・手足の比率を変える | MageFlow deaging v2 |
| [realistic_deaging_slider_fullbody.yaml](realistic_deaging_slider_fullbody.yaml) | 実写風で着衣の成人女性から幼児へ、年齢・顔・頭身・手足の比率を変える | anime deagingの写真調版 |
| [realistic_deaging_slider_fullbody_v2.yaml](realistic_deaging_slider_fullbody_v2.yaml) | 幼児化の記述を保ち、ペア共通の服の色を6組に分散する | realistic deagingの配色改訂版 |
| [realistic_female_only_deaging_slider_v1.yaml](realistic_female_only_deaging_slider_v1.yaml) | 男女ペアで成人女性だけを幼児の女の子へ変える | 既存deagingの年齢・頭身表現を用いた未生成候補 |
| [realistic_female_only_deaging_slider_v2.yaml](realistic_female_only_deaging_slider_v2.yaml) | 成人女性だけの幼児化を、配色・左右を交差した全身12件で学習する | v1の幼児化を維持し、女性aging v2の配色と保持記述を展開した未学習候補 |
| [realistic_male_only_deaging_slider_v1.yaml](realistic_male_only_deaging_slider_v1.yaml) | 男女ペアで成人男性だけを幼児の男の子へ変える | 既存deagingの年齢・頭身表現を用いた未生成候補 |
| [realistic_male_only_deaging_slider_v2.yaml](realistic_male_only_deaging_slider_v2.yaml) | 成人男性だけの幼児化を、配色・左右を交差した全身12件で学習する | v1の幼児化を維持し、女性aging v2の配色と保持記述を展開した未学習候補 |
| [female_size_increase_slider.yaml](female_size_increase_slider.yaml) | 通常サイズの成人男性との構図で、成人女性の全身を比例拡大する | Krea2用の新規プロンプト設計 |
| [female_size_increase_slider_v2.yaml](female_size_increase_slider_v2.yaml) | 通常サイズを比較基準に、very tallで成人女性の比例拡大を狙う | 10候補の画像評価後の未生成改訂案 |
| [female_size_decrease_slider.yaml](female_size_decrease_slider.yaml) | 通常サイズの成人男性との構図で、成人女性の全身を比例縮小する | Krea2用の新規プロンプト設計 |
| [female_size_decrease_slider_v2.yaml](female_size_decrease_slider_v2.yaml) | 通常サイズを比較基準に、very shortで成人女性の比例縮小を狙う | 10候補の画像評価後の未生成改訂案 |

female/male-only aging v2は各**学習用12件（全身4件＋腰上8件）**、female/male-only deaging v2は各**全身12件**、その他の各ファイルは**学習用6件**です。単独人物のaging/deagingは全身4件＋腰上2件、female/male-only aging v1は全身2件＋腰上4件、female/male-only deaging v1は全身6件、breast初版は全身4件＋膝下まで2件、breast v2は頭から腰下まで4件＋全身2件です。clothing fitの既存版はAnimaの学習用0〜5を引き継ぎ、上半身2件＋腰上2件＋腿上2件です。clothing fit改訂版は同じ6種類の衣装・構図をもとに、頭から腰下まで5件＋膝下まで1件へ広げています。参照元の評価用6/7は収録していません。現在のKrea2ノードは渡された全件を学習に使用し、学習・評価indicesの指定を持たないためです。

Female sizeの初版2セットも各6件で、すべて成人男女2人の全身構図です。女性が左の3件と右の3件、正面3件と斜め3件、3種類の衣装を組み合わせています。

## 読み込み方

`Krea2 Slider Train LoRA`の **prompt_yaml** 一覧からファイル名を選択します。ネイティブのMODEL／CLIPも同じ学習ノードへ接続します。テキストの貼り付けは不要です。

ファイルは通常のYAMLブロック形式で保存しています。`>-`による折り返しとYAMLアンカーも読み込めます。今回の形式変更でプロンプト本文は変えていません。JSON互換のYAMLも引き続き読み込めます。追加ファイルはこのフォルダー内の`.yaml`／`.yml`が一覧の対象です。

フィールドは現行ノードに合わせた`target`、`positive`、`negative`、`neutral`だけです。MageFlow／Animaの`unconditional`をKrea2の`negative`へ対応づけています。移植した既存版は`target = negative = neutral`で、`positive`だけに学習したい差分があります。breast v2、clothing fit改訂版、female size初版は`target = neutral`を基準に、`positive`と`negative`を対極の概念にしています。breast v2は中程度の胸部ボリュームを基準に、大きめ／小さめを対比します。`negative`は比較対象の概念であり、生成時の除外語句リストではありません。

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

## 男女ペアの女性限定・男性限定aging v1

女性用は `realistic_female_only_aging_slider_v1.yaml`、男性用は `realistic_male_only_aging_slider_v1.yaml` を選び、それぞれ別のLoRAとして学習します。`output_name`の例は `krea2_female_only_aging_v1` / `krea2_male_only_aging_v1` です。

各6件で、3場面を左右反転した全身2件＋腰上4件です。各人物は左3件・右3件に配置します。`target = negative = neutral`を通常状態とし、`positive`で対象人物の顔・首・手の肌の1文だけを加齢表現に変えています。髪色・髪型・体格・服・姿勢・カメラ・背景と、もう一人の記述はペア内で同一です。対象外の人物を画像上で完全に固定する仕組みではありません。

女性用は[生成確認用TXT](../docs/realistic-female-only-aging-prompt-candidates-v1.txt)の12本文をそのまま移しています。男性用は同じ衣装・背景・構図を保って男性先行の記述にし、肌の加齢対象を男性へ変更しました。対象男性には固定年齢を指定せず、保護対象の女性を30代の滑らかな肌として全roleに共通で記述しています。男性用は新しい文面で、画像生成・選択性の検証は未実施です。

女性用の提供画像35枚では、比較できた17組で女性の加齢が明確で、目視上の明らかな男性の高齢化は認めませんでした。女性の髪・服の裾・体格・姿勢などには変化が残ります。生成条件は `moodyKrea2MixUncensoredV8SEGoes_v80_int8.safetensors`、darkbrush=0.8、1024×1024、8 steps、CFG=1、euler/simple、seed=42/1234/20261015でした。これはdarkbrush併用でのプロンプト生成の評価であり、ベースモデル単体や学習後LoRAでの選択性を実証したものではありません。

提供画像のうち02-Rと03-L/Rの加齢9枚には区切り文字 `END PROMPT` が入り、01-L NORMALのseed=20261015は未収録でした。このYAMLの本文に区切り文字は含めていません。検証時は、女性用ではdarkbrushなしでの代表ペアの再確認、男性用では通常／男性加齢の同一seed比較と左右反転を優先します。

学習は `training_direction=single`、`eta=1`、`teacher_norm_reference=none` を比較基準にします。前回のsize decrease学習実績に揃える場合は `teacher_guidance_scale=1.5`、rank/alpha=16/16、attention、400 steps、learning_rate=1e-4、768×768、trajectory_steps=8です。元のagingの開始値はguidance=1で、この値ではtarget=negativeのため教師がpositive予測と一致します。1.5は差分を1.5倍に外挿する比較条件であり、今回の加齢セットでの最適値は未確認です。これらはノード側で指定し、YAMLから自動設定されません。生成時のCFGとは別の係数です。

2026-10-15に女性用v1の400-step学習レポートと生成画像8枚を確認しました。女性の加齢は強く出ますが、+3では男性の加齢・青緑系の服の明色化・裾の変化が残っています。男性用の学習は未確認です。最新の評価と設定案は[女性限定aging v2の分析](../docs/realistic-female-only-aging-v2-analysis.txt)を参照してください。

## 女性限定aging v2：強度+3での保持を目指す候補

`realistic_female_only_aging_slider_v2.yaml`は12件です。orange/teal、navy blue/burgundy、medium gray/ivoryの3配色を男女で交換し、各組を左右反転しています。各ペアでは女性の皮膚の1文だけを変え、男性・衣服の色と裾・髪・体格・照明は全roleで共通です。`target = negative = neutral`です。

初回の比較条件は、確認済みRAW、`single`、`attention`、rank/alpha=16/16、lr=1e-4、400 steps、768×768、trajectory_steps=8、eta=1、**teacher_guidance_scale=1**、norm=none、seed=42、vary_seed=trueです。設定はノード側で入力します。これは最適性が確認済みの値ではありません。

2026-10-16の学習レポートでは、Moody系モデルで上記の数値設定による400-step学習が完了しています。提供画像9枚（全身0/1/2/3、腰上0/1/2/3/4、各seed=42、darkbrush=0.8）では、女性の加齢と服の色保持を確認しました。+2以降の裾・服の前立て・ポーズ・髪型の変化や、高強度での男性への加齢作用は残っています。評価した2構図は学習プロンプトに含まれ、v1との同条件比較や未学習構図での一般化は未確認です。学習元のRAW由来も、空のsource_metadataからは確定できません。

目標は**+3で女性の加齢量を保ち、男性・衣服への副作用を抑えること**です。弱く効くLoRAに変えるだけでは合格にしません。現行学習は+1のみの全体MSEで、男性・衣服の保持lossはありません。プロンプトだけで+3の無副作用を保証せず、[学習前に作成した分析・比較手順](../docs/realistic-female-only-aging-v2-analysis.txt)に従い評価します。[評価専用6プロンプト](../docs/realistic-female-only-aging-v2-evaluation.json)は学習YAMLに混ぜません。

## 男性aging・男女deaging v2への展開

`realistic_male_only_aging_slider_v2.yaml`、`realistic_female_only_deaging_slider_v2.yaml`、`realistic_male_only_deaging_slider_v2.yaml`は、女性aging v2の方法を展開した各12件の候補です。orange/teal、navy blue/burgundy、medium gray/ivoryの3配色について、男女の色を交換し、左右も反転します。各対象人物は左6件・右6件です。衣服の単色・マットな素材・裾とウエストの記述を対で揃え、`target = negative = neutral`とします。

男性agingは全身4件＋腰上8件です。男性を先に記述し、positiveでは男性の顔・首・手の皮膚の1文だけを加齢表現へ変更します。対象男性の年齢を数値で固定せず、同席する女性を30代の成人として全roleで共通にします。

deagingの2種類は、v1と同じ**成人から幼児への変換**を維持します。対象人物の年齢・顔・頭身・手足の比率のブロックだけを変更し、全12件を同じ床・奥行きの全身構図にします。人数指定は「着衣の2人」とし、対象人物の服と靴は本人の身体に合うサイズとします。同席する成人の年齢・肌・体格、髪、衣服の種類と色、ポーズ、カメラ、背景、照明は各対で固定します。

3種類とも別々のLoRAとして学習します。比較開始条件は女性aging v2に合わせ、`single`、`attention`、rank/alpha=16/16、lr=1e-4、400 steps、768×768、trajectory_steps=8、eta=1、`teacher_guidance_scale=1`、norm=none、seed=42、vary_seed=trueです。`output_name`の例は `krea2_male_only_aging_v2` / `krea2_female_only_deaging_v2` / `krea2_male_only_deaging_v2` です。これらの設定はノード側で入力し、YAMLからは自動適用されません。

この3種類の学習・画像評価は未実施です。女性agingでの改善を他の方向の実証とはせず、強度0/1/2/3・同一seedで、対象の変化量と同席者・衣服の保持を比較します。特にdeagingでは、幼児らしい顔と身体比率、服の適切なサイズ、同席する成人の年齢と画面内サイズが両立するかを確認します。

## 男女ペアの女性限定・男性限定deaging v1

女性用は `realistic_female_only_deaging_slider_v1.yaml`、男性用は `realistic_male_only_deaging_slider_v1.yaml` を選択します。既存realistic deagingと同じく、女性は成人から幼児の女の子へ、男性は成人から幼児の男の子へ変える方向です。成人の範囲内で肌だけを若返らせる設定とは異なります。

各6件で、3場面の左右反転、対象人物が左3件・右3件です。成人と幼児の身長・頭身差が見え、両者が同時に画面に収まるよう、全件を同じ床・奥行きの全身構図にしています。年齢・顔・身体比率の表現は `realistic_deaging_slider_fullbody_v2.yaml` を参考にし、衣装と配色は今回の男女ペアagingの3場面に合わせています。

`target = negative = neutral`は成人2人の通常状態です。`positive`では対象人物の年齢と顔・頭身・手足の比率の記述ブロックだけを変更します。同席する人物は全roleで30代の成人とし、髪・衣装の種類と色・姿勢・背景・カメラはペア内で共通です。服と靴は対象人物の身体に合うサイズを共通に指定します。幼児化後も成立するよう、共通の人数指定は「着衣の2人」で、成人2人や対象人物の成人頭身をpositiveに残していません。

両セットとも `training_direction=single`、`eta=1`、`teacher_guidance_scale=1`、`teacher_norm_reference=none`を開始値にします。正のLoRA強度で幼児化を狙います。設定はノード側で指定し、YAMLからは自動適用されません。別々に学習し、`output_name`の例は `krea2_female_only_deaging_v1` / `krea2_male_only_deaging_v1` です。

どちらも新規の未生成・未学習候補です。女性限定agingで得られた選択性を、deagingの成功実績として扱っていません。通常／幼児化を同一seedで比較し、左右と複数seedで、人数が2人のままか、対象が幼児らしい顔・頭身になるか、同席する成人の年齢・画面内サイズ・衣装が保たれるかを確認します。

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

## Realistic deaging v2 と female size の実機評価候補

`realistic_deaging_slider_fullbody_v2.yaml`は元の6件（全身4件＋腰上2件）の成人／幼児記述、衣服の形、姿勢、構図、背景を維持し、服の配色を変更した学習用ファイルです。各ペアの全roleで衣服と色は同一です。青＋ベージュ、緑＋テラコッタ、コーラル＋ティール、黒ジャケット＋青スカート、黄カーディガン＋アイボリー、ローズの6配色で、暗色の服への偏りを減らします。全roleにcolor photographとnatural, balanced colorsを共通に指定しています。

`target = negative = neutral`、`positive`が幼児化です。前回のdeagingとの比較ではrank/alpha 16/16、400 steps、learning_rate=1e-4、768×768、trajectory_steps=8、eta=1、teacher_guidance_scale=1、teacher_norm_reference=none、training_direction=singleを維持し、配色改訂の影響を確認します。YAMLは学習設定を自動変更しません。白黒化の改善、実トークン数、再学習後の画質は実機で未検証です。

女性サイズの10案は[実機評価用プロンプト集](../docs/female-size-prompt-candidates-v2.txt)にあります。各案に通常・縮小・拡大の全文を収録しています。まず対象Sliderとdarkbrushを外し、同じseedで人数・女性／男性の身長比・男性自身の画面内サイズ・構図を確認します。10案を一括で学習用に採用せず、良好な案を選んでから学習用ペアを作ります。

## Female size v2：生成画像の評価を踏まえた改訂候補

候補10案×3状態のPNG30枚を照合しました。候補02のSMALLで3人化し、ほかの29枚は2人でしたが、指定した半分／2倍を十分に実現した案はありません。06/08の拡大方向が比較的見えることを参考に、女性先行の文面とvery short／very tallによる直接指定を組み合わせたv2を追加しています。新しい組み合わせと英文は未生成で、最適性が確認できたプロンプトではありません。

増加・減少用とも各6件で、3基本場面の左右拡張です。`target = negative = neutral`は通常サイズ、`positive`はそれぞれ拡大／縮小です。`single`、`eta=1`、`teacher_guidance_scale=1`、`teacher_norm_reference=none`を開始値とすると、式上positiveの予測を教師にします。初版の対極差分・guidance=1.5から教師構成も変わるため、文面単独の比較実験ではありません。ほかの設定は前回のrank/alpha 16/16、attention、400 steps、lr=1e-4、768×768、trajectory_steps=8を比較基準にできます。

提供画像はdarkbrush=0.8併用・1 seedでした。学習前にRAWモデルだけでv2の人数、相対身長、男性自身の画面内寸法、成人の頭身、全身の欠けを複数seedで確認してください。検証記録とコピー用のNORMAL／SMALL／LARGE全18本文は[評価・v2プロンプト全文](../docs/female-size-slider-v2-validation.txt)に収録しています。

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
