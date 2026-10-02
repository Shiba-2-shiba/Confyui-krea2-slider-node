# Krea2 女性用 aging v2 生成画像評価の引き継ぎ

> 2026-10-03追記：v2のレポートと `生成画像/生成画像３` を評価した。[評価・対策記録](aging-v2-evaluation.md)を参照。以下は2026-10-02時点の履歴。親フォルダーのv1立位画像と新v2画像は、同強度でLoRA名以外の生成条件が一致し、男性への波及は残った。追加2件にも男性単独anchorを付ける比較用YAMLを用意したが、再学習・改善検証は未実施。

2026年10月2日作成。次のチャットでは、女性用aging v2で学習したLoRAの生成結果を評価する。従来版は女性単独の加齢と男性単独の保持が見られたが、男女同時では男性も加齢した。この問題に対応するため、既存6レコードに男女同時の左右2パターンを追加したv2 YAMLを作成した。**v2の実学習・生成画像評価は、このチャットでは未実施。**

## 作業場所と保存済みの変更

- リポジトリ：`C:\ComfyUI\custom_nodes\Confyui-krea2-slider-node`
- GitHub：<https://github.com/Shiba-2-shiba/Confyui-krea2-slider-node>
- ブランチ：`feat/anchor-gender-preservation`
- push済みコミット：`f523ab81a4d1f8ba72e5084cb1e7799583cc0b00`
- コミット件名：`Improve aging slider prompts and training reliability`
- v2：[prompts/aging_slider_fullbody_v2.yaml](../prompts/aging_slider_fullbody_v2.yaml)
- 使用説明：[prompts/README.md](../prompts/README.md)
- 損失の説明：[docs/anchor-preservation.md](anchor-preservation.md)

生成画像とレポートはGitに追加していない。別マシンへGit cloneしただけでは画像を取得できないため、以下のローカルファイルがあるか確認する。

## 評価目的と重要な訂正

目的は、対象の性別には加齢効果が現れ、反対の性別の顔・首・手や人物の特徴が保たれること。今回の実験とv2は**女性用**であり、女性を加齢させ男性を保持する。

会話の初期には男性用agingの評価プロンプトを相談したが、実際に提供された学習レポートは `aging_slider_fullbody.yaml` の女性用だった。男性用は別ファイル `aging_slider_fullbody_male.yaml`。これらを混同しない。

**`krea2_darkbrush.safetensors` はユーザーによればTurbo用LoRAであり、画風LoRAではない。** 初回解析で画風LoRAと誤認したが訂正済み。既存の比較ではその強度を0.8に固定している。名称から役割を推測し直したり、画風LoRAとして取り外すことを提案したりしない。

## 従来版の学習レポート

ファイル：`C:\ComfyUI\custom_nodes\Confyui-krea2-slider-node\生成画像\krea2_aging_20261002T050755Z_fccfa81a.json`

このレポートは**学習レポート**。個別画像の生成条件はPNGの `prompt` メタデータから確認した。

| 項目 | レポートの値 |
|---|---|
| 学習YAML | `aging_slider_fullbody.yaml`、女性単独6件、男性anchor6件 |
| 学習LoRA | `krea2_aging_20261002T050755Z_fccfa81a.safetensors` |
| steps | 400 |
| rank / alpha | 16 / 16 |
| target | attention |
| learning_rate | 0.0001 |
| 学習解像度 | 768 × 768 |
| trajectory_steps | 8 |
| eta / teacher_guidance_scale | 1 / 1 |
| teacher_norm_reference | none |
| training_direction | single、LoRA +1の学生のみ |
| anchor_strength | 1 |
| seed / vary_seed | 42 / true |
| compute_dtype / quantization | BF16 / convrot_int8 |
| blocks_to_swap / memory_budget_gib | 6 / 14 |
| gradient_checkpointing | true |
| max_grad_norm | 1 |
| ベースモデルの記録 | `/tmp/ComfyUI/models/diffusion_models/intorealismAsian_k2JAVFLASHV1.safetensors` |
| 実験環境の記録 | Linux、NVIDIA RTX A4000、VRAM 15.724GiB、torch 2.13.0+cu130 |
| 状態 | training_completed |
| 学習時間 | 16010.43秒、約4時間27分 |

レポートでは全400ステップで非ゼロの勾配があり、anchorも全ステップで有効だった。Slider損失の平均は最初の100ステップで0.01005436、最後の100ステップで0.00249858。最大grad_normは0.22618で、設定されたクリッピング上限1未満。最大torch reservedは13.38867GiB。

これらは学習処理が進んだ根拠であり、生成画像の性別分離や収束を保証しない。レポートの `source_kind=raw` は設定値で、`source_metadata` は空。実重みのRAW/Turbo区分は独立には確定していない。学習と評価で記録されたベースのファイル名は同じ。

## 従来版の生成条件

以下の2組はいずれも上記の従来版LoRAを使っている。v2の画像ではない。

| 項目 | PNGに記録された値 |
|---|---|
| ベースモデル | `intorealismAsian_k2JAVFLASHV1.safetensors` |
| Turbo用LoRA | `krea2_darkbrush.safetensors`、強度0.8 |
| aging LoRA | `krea2_aging_20261002T050755Z_fccfa81a.safetensors` |
| text encoder | `qwen3vl_4b_fp8_scaled.safetensors`、type=krea2 |
| VAE | `qwen_image_vae.safetensors` |
| seed | 1 |
| steps / CFG | 8 / 1 |
| sampler / scheduler | euler / simple |
| denoise | 1 |
| 解像度 | 1024 × 1024 |
| negative conditioning | ConditioningZeroOut |

各プロンプト内の強度比較は、aging LoRA強度だけを変えていることをPNGメタデータの比較で確認した。

## 最初の生成画像

場所：`C:\ComfyUI\custom_nodes\Confyui-krea2-slider-node\生成画像`

男女が並ぶ立ち姿の腰上構図。手は十分に見えていない。

| ファイル | aging強度 |
|---|---:|
| `Krea2_turbo_00041_.png` | 0 |
| `Krea2_turbo_00042_.png` | 0.5 |
| `Krea2_turbo_00043_.png` | 1 |
| `Krea2_turbo_00044_.png` | 1.5 |
| `Krea2_turbo_00045_.png` | 2 |
| `Krea2_turbo_00046_.png` | 2.5 |
| `Krea2_turbo_00047_.png` | 3 |

目視では強度を上げると男女ともに加齢した。+2以上で目元・ほうれい線・首の変化が明確になり、+3では服の襟や形も変わった。この時点では、単独人物での性別保持を評価できなかった。

## 単独と同時登場を分けた生成画像

場所：`C:\ComfyUI\custom_nodes\Confyui-krea2-slider-node\生成画像\生成画像２`

フォルダー名の末尾は全角の `２`。親フォルダーと同じ画像ファイル名があるため、フルパスで区別する。

| 条件 | 強度0 | 強度1 | 強度2 |
|---|---|---|---|
| 女性単独 | `Krea2_turbo_00041_.png` | `Krea2_turbo_00042_.png` | `Krea2_turbo_00043_.png` |
| 男性単独 | `Krea2_turbo_00044_.png` | `Krea2_turbo_00045_.png` | `Krea2_turbo_00046_.png` |
| 男女同時 | `Krea2_turbo_00047_.png` | `Krea2_turbo_00048_.png` | `Krea2_turbo_00049_.png` |

**観察事実として確信度が高い点**：女性単独は+1で控えめな変化、+2で顔・首・手の明確な加齢が見られた。男性単独は+1でほぼ維持され、+2でも顔の大きな加齢は見られず、手や輪郭の変化は小さかった。男女同時は+2で男性にも目元・ほうれい線・首の加齢が明確に現れた。

**原因の推論として確信度が中程度の点**：単独人物で見られる性別の選択性が、男女同時の構図には十分に一般化していない。従来YAMLは女性単独と男性単独の保持条件だけだったため、この説明と結果が整合する。

anchorなしで学習した比較LoRAがないため、男性単独の保持をanchorの効果だけに帰属させることはできない。また、全画像はseed=1なので、別の人物・seedで同じ挙動になるかは未確認。

## 作成した v2 YAML

ファイル：`C:\ComfyUI\custom_nodes\Confyui-krea2-slider-node\prompts\aging_slider_fullbody_v2.yaml`

元の女性単独6件を全フィールドそのまま複製し、男性anchorも維持した。追加2件はテーブルに並んだ男女の腰上構図で、男性が左／女性が右と、その逆の2パターン。

- `target`：男女ともに滑らかな成人の肌。
- `positive`：男性の肌の記述を維持し、女性の顔・首・手だけを加齢させる。
- `negative` と `neutral`：`target`と同じ。YAMLの参照構文で共用する。
- 追加2件の `anchor`：省略。男女とも基準状態を保持する損失を追加すると、目的の女性加齢も抑える方向になるため。

targetとpositiveで、女性の加齢以外に人物配置・髪・服装・表情・背景・照明の差がないことをテストで確認した。既存ローダーが7ファイル目として検出し、全8レコードを読み込める。トレーナーのコード変更は不要。

これは人物領域をマスクして保持する実装ではなく、テキスト条件の差分を追加する変更。男女同時の保持改善はまだ仮説であり、保証ではない。追加したテーブル構図は今後のv2では学習条件になるため、同じ構図の改善と、未学習の背景・服装・姿勢への一般化を分けて評価する。

ノードは全レコードを順番に使用する。400ステップでは6件版が各66〜67回、8件版が各50回になる。総ステップ数と各レコードの学習回数のどちらをそろえた比較なのかを記録する。

## ここまでのコード修正と検証

push済みコミットには、v2 YAML・説明・回帰テストのほか、次の修正が含まれる。

- VRAM予算のUIと設定検証から14GiBの固定上限を撤廃。既定14GiB、最小1GiB、有限値の検証は維持。実際のallocator上限には空きVRAMと既存上限も反映される。
- GPU後始末が例外になっても学習ロックを解放する。
- 既存YAMLの固定ハッシュ検査でCRLFをLFに正規化し、Windowsの改行変換による失敗を解消する。

コミット直前の全テスト：`python -B -m pytest -q` → **54 passed, 1 skipped, 79 subtests passed**。スキップは同梱されていない公式参照コードとのモデル一致テスト。新YAMLの読込、既存6件の一致、追加2件の女性のみの差分、左右反転、選択一覧への検出を確認済み。これはv2の画像品質検証ではない。

最初のコード解析では、LoRA確定後にJSONレポートの確定が失敗するとLoRAだけが残る保存処理も指摘したが、今回の修正対象には含めていない。

## 次のチャットでの評価手順

1. ユーザーから新しい生成画像フォルダーと学習レポートを受け取り、実際にv2 YAMLで学習したか、LoRAファイル名と設定を確認する。古いレポートや同名PNGと混同しない。
2. PNGメタデータから実際のLoRA強度・seed・モデル・Turbo LoRA・sampler・scheduler・steps・CFG・解像度・プロンプトを確認する。ファイル名の番号だけで強度を推測しない。
3. 女性単独・男性単独・男女同時の各条件で、0／1／2を比較する。女性の顔・首・手の加齢と、男性の加齢・顔立ちの変化、衣服・姿勢・構図への波及をそれぞれ見る。
4. v1に対して、女性への効果を維持しながら、男女同時の男性への波及が減ったかを判断する。男性保持だけ改善して女性の効果も消えた場合は、目的を達成したと扱わない。
5. 可能なら左右反転、別seed、未学習の背景・衣装・姿勢も評価する。画像差分の大きさや学習lossだけを、加齢や性別分離の指標として扱わない。
6. 観察事実・原因の推論・未確認事項を分けて報告する。今回の目的は生成結果の評価なので、依頼なしにコード変更や再学習を開始しない。

実用上の目標は、女性単独と男女同時の女性で加齢が現れ、男性単独と男女同時の男性が保たれること。強度1での効果の弱さと、強度2以上での波及を別々に評価する。

## 次のチャットへ貼る開始文

```text
このリポジトリの docs/handoff-aging-v2-evaluation.md を読み、女性用aging v2 LoRAの生成結果を評価してください。
従来版は女性単独の加齢と男性単独の保持が見られましたが、男女同時では男性にも加齢が波及しました。
v2は既存6件を維持し、女性だけを加齢させる男女同時の左右2パターンを追加しています。
krea2_darkbrushはTurbo用LoRAで、画風LoRAではありません。
新しい画像と学習レポートの場所は、この後指定します。
```
