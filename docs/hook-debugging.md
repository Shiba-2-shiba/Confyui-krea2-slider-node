# 領域Hookの診断ログ

このログは、マスクとHookがどのconditioningに付いたか、INT8重みのHook切替後にバックアップが残っていないかを調べるためのものです。プロンプト本文、埋め込みテンソル、重み名や重み値は記録しません。診断が正常に動作する場合、CONDITIONINGは変更せず通します。診断そのものが失敗した場合は、原因を隠さずスタックトレースを出して停止します。

## 再現手順

1. ComfyUIにこのカスタムノードの更新を読み込ませ、[デバッグ用ワークフロー](../workflows/krea2_two_person_region_slider_debug.json)を開きます。
2. モデル・CLIP・VAE・LoRAのファイル名を自分の環境に合わせます。このテンプレートでは胸Sliderを`0`、deaging Sliderだけを`2`にし、左マスクを`x=[0,472), y=[376,1024)`相当に縮めています。胸と幼児化のLoRAは同時に有効にしないでください。
3. `Krea2 Native LoRA Hooks Fix`の`debug_logging=true`、`Krea2 Conditioning Debug`の`enabled=true`を確認して、1枚生成します。
4. ComfyUIを起動したコンソールの`[Krea2HookDebug]`行を保存します。フルマスクと小マスクを比較するときはseed、モデル、プロンプト、LoRA強度を固定し、マスクだけを変えます。

## 出力の読み方

1行は`[Krea2HookDebug]`に続くJSONです。複数Samplerを同時実行すると行が混在するため、診断時は1件ずつ生成してください。
`patcher_id`、`model_id`、`group_id`、`hook_ref_id`は、このComfyUIプロセス内でだけ照合できる不透明なIDです。

| `event` | 調べられること |
| --- | --- |
| `conditioning` | Samplerに渡す直前のpositive/negative各条件。`mask.bounds_xyxy`は右端・下端を含まない画素座標、`mask.coverage`は非ゼロ率、`default`は残余領域用条件、`hooks.items[].strength_model`は接続されたLoRA強度です。`uncovered_fraction`は矩形マスクの幾何学的な空き率で、Sampler内部の重みと同一ではありません。 |
| `hook_fix_installed` | 互換修正を付けたpatcherと共有内部モデルのID。`dynamic`はDynamicVRAMの経路を示します。 |
| `hook_switch_start` / `hook_switch_end` | SamplerのHookGroup切替前後。Hookなし（`requested.present=false`、または`count=0`）への切替後に`backup_after=0`なら、重みバックアップは残っていません。 |
| `hook_restore` | 実際に戻した重みの件数と残数。`backup_after=0`、`current_hooks_cleared=true`を確認します。 |
| `hook_switch_error` | Hook切替中の例外。`error_type`だけを記録します。 |

小マスクの期待値は、女性条件が`bounds_xyxy=[0,376,472,1024]`かつHookあり、男性条件が右半分・Hookなし、背景のDEFAULTがHookなしです。これと異なる場合はワークフロー接続またはマスク入力を調べます。

Hookなし条件への切替後も`backup_after>0`なら、カスタムノードの復元経路を調査する根拠になります。すべて復元され、条件メタデータも正しければ、境界上の二重女性は複数の全画像予測が硬い矩形境界で食い違う生成挙動という仮説が強まります。ただしログは最終画素の原因を単独で証明しません。

ログは`debug_logging=false`と診断ノードの`enabled=false`に戻すと停止します。無効時は診断ノードが同じ入力に対して安定したfingerprintを返すため、再実行を強制しません。既存の通常ワークフローは診断ノードを含みません。
