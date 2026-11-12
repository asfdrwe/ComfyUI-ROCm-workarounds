# ComfyUI ROCm Workarounds

ComfyUI の AMD/ROCm 環境向けに、2つの回避策を起動時に適用する小さなカスタムノードです。

ChatGPT に作ってもらいました。

ComfyUI 本体の `comfy/sd.py` や `comfy/ops.py` を直接編集せずに済ませられ、本体を頻繁に `git pull` しても、ローカルの手修正による競合を避けられます。

ノードUI、ワークフローへのノード追加、設定ファイル、追加の Python パッケージは必要ありません。

## 変更内容

### 1. VAE のメモリ使用量推定を補正する

対象: `comfy/sd.py` の `VAE` クラス

ComfyUI 本体では、AMD 環境の VAE メモリ推定に `VAE_KL_MEM_RATIO = 2.73` が使用されています。このカスタムノードは、`VAE.__init__` の呼び出し後に、該当する `memory_used_encode` と `memory_used_decode` の推定値を補正し、**係数 `1.0` 相当**にします。

### 2. 特定の Conv3d を `F.linear` で計算する

対象: `comfy/ops.py` の `disable_weight_init.Conv3d._conv_forward`

[ComfyUI PR #17166](https://github.com/Comfy-Org/ComfyUI/pull/17166) を参考に、**Conv3d のカーネルが入力の空間・時間方向全体を覆う場合**、通常の3D畳み込みの代わりに `torch.nn.functional.linear` を使用します。

適用条件は次のとおりです。

- 入力とカーネルの空間・時間方向のサイズが一致する（`causal_zero` 時はスライス後の重みで判定）
- パディングなし
- dilation がすべて `1`
- groups が `1`

条件に当てはまらない場合は、ComfyUI 本来の `_conv_forward` を呼び出します。`manual_cast.Conv3d` も対象クラスを継承しているため、この置き換えの影響を受けます。

## インストール

custom_nodes フォルダで `git clone https://github.com/asfdrwe/ComfyUI-ROCm-workarounds/`してください。

AMD/ROCm 環境で `comfy.model_management.is_amd()` が `True` を返す場合だけ、読み込み時にパッチを適用します。それ以外の環境では何もしません。

## 使い方

インストールして ComfyUI を起動するだけです。ワークフロー上に追加するノードはありません。

パッチが登録された場合、起動ログに次のようなメッセージが表示されます。

```text
[ROCm] VAE memory ratio workaround installed
[ROCm] Conv3d workaround installed
```

Conv3d 側で本体に `linear` の実装があると簡易判定された場合は、代わりに次が表示されます。

```text
[ROCm] Conv3d workaround skipped
```

## ComfyUI の更新

ComfyUI 本体に変更を加えないため、通常どおり `git pull` できます。

ただし、**ComfyUI 内部のクラス・メソッド・推定関数の構造が変わった場合には、このカスタムノード側の修正が必要になることがあります。** 本体の更新後に起動エラーや動作の変化があれば、まずカスタムノードを外して切り分けてください。

Conv3d 側の「本体に `linear` があるかどうか」という判定は、メソッドのコード内に名前が存在するかを見る簡易的なものです。上流の修正を完全に検出できる仕組みではありません。

## アンインストール

ComfyUI を終了し、`custom_nodes/rocm_workarounds` を削除してください。その後 ComfyUI を再起動します。

## ライセンス

CC0 1.0 なので自由にどうぞ。

## 参考リンク

- [ComfyUI 本体](https://github.com/Comfy-Org/ComfyUI)
- [ComfyUI PR #17166 — Use linear for Conv3d when the kernel covers the input](https://github.com/Comfy-Org/ComfyUI/pull/17166)
- [patientx-cfz/comfyui-rocm](https://github.com/patientx-cfz/comfyui-rocm)
