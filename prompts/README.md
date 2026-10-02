# Krea2 Slider prompts

Krea2用の英語の自然文で、人物・画風・衣装・構図・背景・光を記述しています。既存の女性用aging/deagingと胸サイズv1/v2の`target`・`positive`・`negative`・`neutral`本文はそのままです。女性用のファイル名も変えていません。

| ファイル | 正のLoRA強度の方向 | anchorで保持する対象 |
|---|---|---|
| [aging_slider_fullbody.yaml](aging_slider_fullbody.yaml) | 成人女性の顔・首・手の皮膚を加齢させる | 同じ構図・衣装の成人男性 |
| [aging_slider_fullbody_v2.yaml](aging_slider_fullbody_v2.yaml) | 女性単独＋男女同時で女性だけの顔・首・手の皮膚を加齢させる | 既存6件の男性anchorを維持。追加の男女同時2件にはanchorなし |
| [aging_slider_fullbody_v2_anchor_ablation.yaml](aging_slider_fullbody_v2_anchor_ablation.yaml) | v2と同じ8件・同じSlider本文。保持損失の比較実験用 | 追加2件にも男性単独anchorを付け、全stepで保持損失を有効にする。改善は未検証 |
| [aging_slider_fullbody_male.yaml](aging_slider_fullbody_male.yaml) | 成人男性の顔・首・手の皮膚を加齢させる | 同じ構図・衣装の成人女性 |
| [deaging_slider_fullbody.yaml](deaging_slider_fullbody.yaml) | 着衣の成人女性から幼児の女児へ、年齢・顔・頭身・手足の比率を変える | 同じ構図・衣装の成人男性 |
| [deaging_slider_fullbody_male.yaml](deaging_slider_fullbody_male.yaml) | 着衣の成人男性から幼児の男児へ、年齢・顔・頭身・手足の比率を変える | 同じ構図・衣装の成人女性 |
| [breast_size_slider_v2.yaml](breast_size_slider_v2.yaml) | 着衣の成人女性の胸を大きくする。小さい胸との明示的な対比 | 同じ構図・衣装で自然な男性の胸部を持つ成人男性 |
| [breast_size_slider.yaml](breast_size_slider.yaml) | 旧版。着衣の成人女性の胸部ボリュームを中程度から大きめへ | なし。ファイルは変更せず互換性を維持 |

aging v2とanchor比較版は**学習用8件**、その他のファイルは**学習用6件**です。既存aging/deagingは全身4件＋腰上2件、breast v1は全身4件＋膝下まで2件、breast v2は頭から腰下まで4件＋全身2件です。参照元の評価用レコードは収録していません。ノードは選択したファイルの全件を学習に使います。

女性用aging v2は、元の6件を全フィールドそのまま引き継ぎ、男女がテーブルに並ぶ腰上構図を2件追加しています。男性が左／女性が右と、その逆の2パターンで、`target`から`positive`への差分は女性の皮膚の加齢だけです。追加2件は`target = negative = neutral`で、男女とも基準状態を保持する損失が女性の加齢と競合しないよう`anchor`を省略しています。既存6件の男性anchorは引き続き有効です。

ノードは8件を順番に使用するため、同じ総step数では各レコードの学習回数が6件版より少なくなります。男女同時での性別分離は下記の立位・seed=1評価で未達成で、他の条件への一般化は未検証です。追加した構図そのものに加え、学習にない服装・背景・姿勢と複数seedで評価してください。

2026-10-03のv2評価では、立位・seed=1で男性への加齢が残りました。anchor比較版は全Slider本文を保ち、追加2件に男性単独anchorだけを付けた実験用です。これは男女同時の画像内の男性をマスクして保持する機能ではありません。最初はv2と同じ400steps・全設定で比較し、女性への効果と男性保持の両方を判定します。再学習前の切り分けと未送信API条件は[評価記録](../docs/aging-v2-evaluation.md)を参照してください。

男性用aging/deagingでは性別を表す語と代名詞だけを変えています。概念の差分に別の変化を混ぜないよう、髪・衣装・構図・光を保持し、元のスカートやブラウスもそのままです。deagingは成人から着衣の幼児へ変える意図を保持しています。若い成人への変更ではありません。

## 読み込み方とフィールド

`Krea2 Slider Train LoRA`の **prompt_yaml** 一覧からファイル名を選択し、ネイティブのMODEL／CLIPを同じ学習ノードへ接続します。追加ファイルはこのフォルダー内の`.yaml`／`.yml`が一覧の対象です。

通常のYAMLブロック形式、`>-`による折り返し、YAMLの`&name`/`*name`参照、JSON互換YAMLを読み込めます。保持用の`anchor`フィールドとYAMLの参照構文は別の機能です。

- `target`：Sliderの基準となるプロンプト
- `positive`：正のLoRA強度で向かう概念
- `negative`：比較対象の概念。生成時の除外語句リストではありません
- `neutral`：任意。教師正規化を`neutral`にしたときに使い、省略時は`target`
- `anchor`：任意の空でない文字列。LoRAによる変化を抑えたい対象。ベースモデルの予測を保持する追加損失に使います

aging/deagingとbreast v1では`target = negative = neutral`です。**breast v2は`target = neutral`が中程度、`positive`が大きい胸、`negative`が小さい胸**で、三者の差分を維持しています。男性anchorを追加するために`negative`を基準へ戻す処理はありません。

Krea2のtext encoderはsystem/userテンプレートを内部で付加します。ファイル内にはチャット用の特殊トークン、system指示、品質スコアタグを入れていません。

## ノード側の設定

全セットで **training_direction=single** を開始値にします。positiveへ向かう方向をLoRA強度`+1`で学習します。`negative`は教師の比較対象で、逆方向の学生学習を意味しません。agingの比較基準は成人の肌で、幼児プロンプトは使いません。

| セット | eta | teacher_guidance_scale | teacher_norm_reference | anchor_strength |
|---|---:|---:|---|---:|
| aging（女性・男性） | 1 | 1 | none | 1 |
| breast size v2 | 1 | 2 | none | 1 |
| deaging（女性・男性） | 1 | 1 | none | 1 |
| breast size v1 | 1 | 2 | none | 効果なし（anchorなし） |

詳細設定の`anchor_strength`は有限の非負値で、既定は`1.0`です。`0`ならanchorのエンコード・教師予測・学生forward/backwardを行わず、従来のSlider損失だけで学習します。`anchor`を省略した独自YAMLにも追加の処理はありません。正の値で有効にすると、元のSlider損失に`anchor_strength × anchor_loss`を足します。Slider損失の重みを半分にする処理ではありません。[保持損失の仕様・レポート](../docs/anchor-preservation.md)

これらは開始値であり、最適値や画質を保証しません。胸サイズの効果が強すぎる場合は`teacher_guidance_scale=1`も比較対象になります。anchorを強めると対象外への波及だけでなく、本来のSlider効果も弱まる可能性があるため、画像で両方を比較してください。

16GB向けの動作確認は512×512、rank/alpha 8/8、INT8、checkpoint ON、CPU退避16ブロックから開始できます。参照元の解像度や`guidance_scale`、`action`、`batch_size`をYAMLから自動適用する処理はありません。anchor有効時は処理時間とメモリをあらためて測定してください。

## 確認範囲と評価

テストで全8ファイルの読込、既存6ファイルの6レコード、aging v2の8レコードと元の6件の完全一致、男女同時2件で女性の加齢以外が変わらないこと、anchor比較版がSlider本文を変えず男性単独anchorだけを追加すること、性別ごとの対応、breast v2の小さい胸の比較対象、breast v1の未変更を検証します。過去のローカルComfyUIのトークン数測定は女性用aging/deagingとbreast v1が対象で、新しい男性用・anchor・aging v2・breast v2全体のトークン数検証の代わりにはなりません。実encoderでも長さを確認してください。

**anchorは対象外への影響がゼロになる保証ではありません。** 実GPUで学習し、男性・女性それぞれを同じseedの`-1 / 0 / +1`画像で比較する必要があります。強度`0`と`1`の学習比較、学習にない衣装・構図も含め、RAWとTurboを別々に評価します。旧版の動作確認やCPUテストだけで性別分離・衣装保持・Slider画質が確認できたとは扱いません。[画像評価の手順](../docs/anchor-preservation.md#required-image-evaluation)

## 出典

女性用aging/deagingとbreast v1は次のMageFlowプロンプトの学習用0〜5を引き継いでいます。

参照元ディレクトリ：`C:/ComfyUI/custom_nodes/Comfyui-mageflow-slider-node/prompts/`

- `prompts-mageflow-aging_slider_fullbody_v2.yaml`
- `prompts-mageflow-breast_size_slider_v5.yaml`
- `prompts-mageflow-deaging_slider_fullbody_v2.yaml`

breast v2の意図と対比は[分析記録](../docs/breast-size-slider-v2-analysis.txt)を参照してください。Krea2の記述方針は調査時の`参考/krea-2/docs/prompting.md`、入力テンプレートと最大長は`参考/krea-2/encoder.py`を根拠にしています。これらの参考ディレクトリはリポジトリに同梱していません。公式のTurbo生成例は、RAW Slider学習での画質実証とは区別しています。
