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

- Baseは撮影様式、カメラ、光、フレーミングだけにします。人物、性別、人数は書きません。
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
