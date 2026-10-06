# Krea2 Regional Attention：実機Gate Aの使い方

[ワークフロー](krea2_two_person_attention.json)は、1回のKSamplerで女性・男性・背景の文章を別々のtoken領域へ割り当てます。最初の目的は、Slider LoRAを使わずに余分な女性、二重の顔、境界で切れた身体が減るかを確認することです。

## 接続と初期値

- `darkbrush`は全体へ掛ける高速化LoRAなので、通常の`LoraLoaderModelOnly`でRegional Attentionより前に接続しています。
- `Krea2 Regional Prompt Region`は女性を先、男性を後に連結しています。マスクが重なるtokenは先の女性regionが優先され、どちらにも属さないtokenは背景へ割り当てられます。
- `Krea2 Apply Regional Attention`は`strict`、診断ログはONです。フェザー、grow、area、native Hook、地域SliderはこのGate A版では使いません。
- KSamplerはseed 42、8 steps、CFG 1、Euler/simpleです。比較中はこの条件を固定します。

## マスクを変える際の注意

初期の女性マスクは`x=0, y=0.367556, w=0.460948, h=0.632444`、男性マスクは右半分全高です。女性マスクを小さくした過去条件を、attention経路を変えた今回の方式で再評価する設定です。

矩形は最終画像の切り抜き枠ではありません。latentをKrea2のpatch単位へ縮小したtoken ownerを決めるため、境界は数pixel単位で完全には一致しません。矩形を動かした後は、診断ログのtoken grid、owner別token数、owner境界を確認してください。MASKの値・形・画像サイズはApply時に検査され、token gridへ縮小した後に1 tokenも残らない領域や先の矩形に完全に隠れた領域は最初のforwardで検出されます。どちらも無制限生成へ切り替えず、エラーで停止します。

女性の頭から胴体までを小さな領域へ収める比較では、文章中の`compact full figure`と`clear margin at every edge`を残します。矩形をさらに縮める場合は、人物文章をそのままにしてマスクだけ変えると、人物そのものが成立しにくくなります。まず上下左右のどの境界を変えたかを1つずつ記録してください。

## プロンプトの分け方

- この初期比較ではBaseを共通の撮影条件にしています。人物をBaseへ書かないことは分離試験の初期条件であり、配置を制御する普遍的な規則ではありません。位置を指定するBaseは、下の`layout_base`で別途比較します。
- Backgroundは壁・床・光だけにします。`woman`、`man`、`person`などを入れません。
- 女性・男性のregion promptには、それぞれ1人だけを書きます。`left half`や`right half`のような固定座標は使わず、`within her/his own assigned frame`として矩形編集と矛盾しにくくしています。
- Gate Aでは胸、年齢、幼児化などのSlider評価語を入れません。まず文章だけの隔離を評価します。

Base textは全領域が共有する撮影条件として読めますが、Base自身は画像tokenを読み戻しません。各regionと背景は、同じownerの文章・画像とBase textだけを参照します。これにより、背景や別regionを中継して女性情報がマスク外へ回る経路も遮断します。

## Gate Aの生成と記録

小マスクと左半分全高マスクを用意し、それぞれseed `42, 123, 777, 2026, 31415`で計10枚生成します。各画像で次を記録します。

1. 成人女性1人と成人男性1人だけか。
2. 女性マスク外に別の顔、頭、身体の断片がないか。
3. 頭と胴体が境界で二重化していないか。
4. 指定領域からのずれが、ログに出た実効token境界1個分以内か。

PNGのAPIメタデータと、同じrun IDの`[Krea2HookDebug] regional_*`ログを一緒に保存してください。10枚すべてが条件を満たしてから、女性だけにSlider LoRAを掛ける次段階へ進みます。人物重複が残る場合は、LoRA強度やプロンプト語を増やさず、owner数・segment長・forbidden edge・attention call数のログで経路を確認します。

## 現段階の制限

このJSONはprompt-onlyの実験版です。Regional Slider、`balanced` release、動画、reference latentには対応していません。`Krea2 Native LoRA Hooks Fix`、`Create Hook LoRA`、`Cond Pair Set Props`、`Krea2 Pair Region Area`を同じMODEL/conditioning経路へ追加しないでください。

許可行列、複数層での情報遮断、V3スキーマ、JSONリンクはCPUテスト済みです。一方、開発側のローカル環境では実機と同じComfyUI commitの読み込みに必要な`comfy_aimdo.storage`が不足しており、Krea2本体を使う統合テストとGPU生成は未実施です。このワークフローの画像品質は、上記Gate Aの10枚と同じrun IDのログで判定してください。

## 初回実機結果と切り分け用JSON

`attention_00001`（seed42）、`00002`（4444）、`00003`（444444444）は同じ小マスク・文章・8 steps・CFG1です。3枚とも女性の頭がマスク上端で切れ、上側にはそれぞれ別の女性が1人／2人／0人見えます。小マスク条件のGate Aは未達です。画像だけでは女性文章が背景へ伝わったかを確定できません。

今回のマスク上端は画素で約376pxですが、token ownerの包含矩形は`[0,23,30,64]`、画像換算で`[0,368,480,1024]`です。画像の水平な切れ目はこの実効境界と一致します。マスクを人物の配置・縮尺へ変換する処理はまだなく、`own assigned frame`という語にも矩形座標は付いていません。

同じseed42、同じモデルで次の3つを個別に比較します。診断ログはONです。ログONには`comfy.__file__=None`の修正版が必要です。

| JSON | 変更する条件 | 確認すること |
| --- | --- | --- |
| [neutral_base](krea2_two_person_attention_neutral_base.json) | Baseを光・色・露出・画質だけにする | 上側の別女性が減るか。人物向け構図語が背景へ与える影響 |
| [layout_base](krea2_two_person_attention_layout_base.json) | Baseへ下左の矩形位置・小さい成人女性・右男性・上左の空壁を明記 | 頭を含む全身が小矩形へ入るか。共有Baseによる領域外人物の再発も評価 |
| [half_mask](krea2_two_person_attention_half_mask.json) | 元の文章のまま女性マスクを左半分全高へ変更 | 上端の制約を外すと女性の頭が戻るか |

既定JSONは比較用に保持しています。これらは原因判別用で、配置改善を検証済みの修正版ではありません。まずseedを固定し、比較する条件以外を変えないでください。`layout_base`の座標文は現在の小矩形用なので、矩形を変更した場合は文も更新します。

参照実装のREADMEは、Baseへの位置ヒントが人物配置を導き、attention maskは分離を担当すると説明しています。また、画像間attentionを終始遮断すると貼り合わせ状の境界が出ることも説明しています。後半で画像間attentionを開く方法と、領域内位置座標へ変換する方法は追加実験の候補ですが、現在は未実装であり、人物重複が解決すると保証できません。
