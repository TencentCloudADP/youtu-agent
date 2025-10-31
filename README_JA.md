# <img src="docs/assets/logo.svg" alt="Youtu-agent Logo" height="24px"> Youtu-Agent：シンプルで強力なエージェントフレームワーク

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-文档-blue.svg></a>
<!-- <a href=https://arxiv.org/abs/2502.14345><img src=https://img.shields.io/badge/arXiv-2502.14345-b31b1b.svg></a> -->
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-腾讯-blue.svg></a>
<a href=https://deepwiki.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/DeepWiki-Tencent-blue.svg></a>
<a href=https://arxiv.org/abs/2510.08191><img src=https://img.shields.io/badge/arXiv-2510.08191-b31b1b.svg></a>
</div>

<p align="center">
| <a href="README.md"><b>英語</b></a>
| <a href="#-ベンチマークパフォーマンス"><b>🌟 パフォーマンス</b></a> 
| <a href="#-使用例"><b>💡 サンプル</b> </a> 
| <a href="#-特徴"><b>✨ 機能</b> </a> 
| <a href="#-すぐに始める"><b>🚀 クイックスタート</b> </a> 
| 📢 <a href="https://discord.gg/QjqhkHQVVM"><b>Join Discord</b></a> or <a href="https://github.com/user-attachments/assets/354cd8e7-e108-4348-9355-04440052f408"><b>WeChat</b></a> 
|
</p>

`Youtu-Agent`は、自律エージェントを構築・実行・評価するための柔軟で高性能なフレームワークです。ベンチマークテストでトップクラスの成績を収めるだけでなく、オープンソースモデルを活用してデータ分析、ファイル処理、深層学習などの高度な機能を実現できる強力なエージェント機能も備えています。

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

主なハイライト：
- **パフォーマンスの検証**：WebWalkerQAではpass@1で71.47%、GAIA（純テキストサブセット）ではpass@1で72.8%を達成しました。これは`DeepSeek-V3`シリーズのモデルのみを使用しており（ClaudeやGPTは使用していません）、強力なオープンソースの出発点を築きました。
- **オープンソースに優しく、コストに敏感**：アクセスしやすく、低コストでのデプロイを最適化しており、クローズドなモデルに依存しません。
- **実際の使用例**：CSV分析、文献レビュー、個人ファイルの整理、ポッドキャストやビデオの生成などのタスクを箱から出してすぐにサポートします。（近日公開予定）
- **柔軟なアーキテクチャ**：[openai-agents](https://github.com/openai/openai-agents-python)に基づいて構築されており、`DeepSeek`から`gpt-oss`までのさまざまなモデルAPI、ツールの統合、フレームワークの実装と互換性があります。
- **自動化とシンプルさ**：YAMLベースの設定、自動エージェント生成、簡素化された設定により、手動の作業負担が減ります。

## 🗞️ ニュース

- 🚀 [2025-10-10] [**Training-Free Group Relative Policy Optimization**](https://arxiv.org/abs/2510.08191)。DeepSeek-V3.2 のRLが8ドルで？本当に実現しました！トレーニング不要の GRPO が DeepSeek-V3.2 を凍結したまま約100件のサンプルからトークン事前分布を学習し、RL のコストをおよそ8ドルまで抑えつつ数学とウェブ検索タスクで改善を確認しました！コードブランチ [training_free_GRPO](https://github.com/TencentCloudADP/youtu-agent/tree/training_free_GRPO) [[X ポスト](https://x.com/cai_cecilia47/status/1976558824640393559)]。
- 🛠️ [2025-09-28] エージェント自動生成に加えて、要件を一度伝えるだけで`Youtu-Agent`がツールを構築してくれる自動ツール生成機能をリリースしました。[[詳細](https://tencentcloudadp.github.io/youtu-agent/auto_generation/)]。
- 📺 [2025-09-09] 「Youtu-Agent」の設計思想と基本的な使い方についてライブ配信を行いました。[[動画](https://www.bilibili.com/video/BV1mypqz4EvS)] [[資料](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)]。
- 🎁 [2025-09-02] [テンセントクラウド国際サイト](https://www.tencentcloud.com/)では、DeepSeek APIの新規ユーザーに対して**300万枚の無料トークン**を提供しています（**2025年9月1日から2025年10月31日まで**）。`Youtu-Agent`でDeepSeekモデルを使用したい場合は、[無料トライアルをクリック](https://www.tencentcloud.com/document/product/1255/70381)してください！企業向けのエージェントソリューションについては、[エージェント開発プラットフォームADP](https://adp.tencentcloud.com)もご覧ください。
- 📺 [2025-08-28] 新しくリリースされたDeepSeek-V3.1モデルについてライブ配信を行い、`Youtu-Agent`フレームワークでの使用方法を紹介しました。[[動画](https://www.bilibili.com/video/BV1XwayzrETi/)] [[資料](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)]。

## 🌟 ベンチマークパフォーマンス

`Youtu-Agent`はオープンソースモデルと軽量なツールをベースに構築されており、チャレンジングなディープサーチやツール使用のベンチマークテストで優れたパフォーマンスを発揮しています。

- **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA)**：`DeepSeek-V3-0324`を使用して60.71%の精度を達成し、新しくリリースされた`DeepSeek-V3.1`を使用すると71.47%に向上し、新たなSOTA（State of the Art）のパフォーマンスを記録しました。
- **[GAIA](https://gaia-benchmark-leaderboard.hf.space/)**：`DeepSeek-V3-0324`（ツールで使用されているモデルを含む）を使用して、[純テキスト検証サブセット](https://github.com/sunnynexus/WebThinker?tab=readme-ov-file#benchmarks)でpass@1で72.8%を達成しました。多モーダルツールを含む完全なGAIAベンチマークの評価を積極的に拡大しており、近日中に完全なトレースを公開予定ですので、ご期待ください！✨

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 使用例

<table border="1" style="border-collapse: collapse;">
  <tr>
    <td style="border: 1px solid black; padding: 10px;">
      <a href="https://www.youtube.com/watch?v=r9we4m1cB6M">
        <img src="https://img.youtube.com/vi/r9we4m1cB6M/0.jpg" alt="データ分析" width="420" height="236">
      </a>
      <br><strong>データ分析</strong><br>CSVファイルを分析し、HTMLレポートを生成します。
    </td>
    <td style="border: 1px solid black; padding: 10px;">
      <a href="https://www.youtube.com/watch?v=GdA4AapE2L4">
        <img src="https://img.youtube.com/vi/GdA4AapE2L4/0.jpg" alt="ファイル管理" width="420" height="236">
      </a>
      <br><strong>ファイル管理</strong><br>ユーザーのためにローカルファイルの名前を変更したり分類したりします。
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px;">
      <a href="https://www.youtube.com/watch?v=vBddCjjRk00">
        <img src="https://img.youtube.com/vi/vBddCjjRk00/0.jpg" alt="広範な研究" width="420" height="236">
      </a>
      <br><strong>広範な研究</strong><br>大量の情報を収集して総合的なレポートを作成し、Manusの機能を再現します。
    </td>
    <td style="border: 1px solid black; padding: 10px;">
      <a href="https://www.youtube.com/watch?v=v3QQg0WAnPs">
        <img src="https://img.youtube.com/vi/v3QQg0WAnPs/0.jpg" alt="論文分析" width="420" height="236">
      </a>
      <br><strong>論文分析</strong><br>指定された論文を解析し、分析を行い、関連する文献を整理して最終結果を出します。
    </td>
  </tr>
</table>

> [!NOTE]
> 詳細については、[`examples`](./examples)ディレクトリと[ドキュメント](https://tencentcloudadp.github.io/youtu-agent/examples/)を参照してください。

### 🤖 自動ツール＆エージェント生成

`Youtu-Agent`の大きな特長は、**エージェント設定だけでなくツールも自動生成**できる点です。他のフレームワークでは、エージェントを動かす前に関数を実装したりプロンプトを細かく設計したりする必要がありますが、Youtu-Agentではタスクを説明するだけで済みます。内蔵のメタエージェントが対話を通じて要件を整理し、必要なツールを組み立て、YAML設定を生成して保存するので、そのまま実行に移せます。

```bash
# インタラクティブに要件を明確にし、設定を自動生成します
python scripts/gen_simple_agent.py

# 生成された設定を実行します
python scripts/cli_chat.py --config generated/xxx
```

<table border="1" style="border-collapse: collapse;">
  <tr>
    <td style="border: 1px solid black; width:420px; padding:10px; vertical-align:top;">
      <a href="https://www.youtube.com/watch?v=JVpHDJtKBo8">
        <img src="https://img.youtube.com/vi/JVpHDJtKBo8/0.jpg" alt="自動エージェント生成" width="420" height="236">
      </a>
      <br><strong>自動エージェント生成</strong><br>インタラクティブな対話によって要件を捉え、エージェントの設定を自動生成し、すぐに実行します。
    </td>
    <td style="border: 1px solid black; width:420px; padding:10px; vertical-align:top;">
      <video src="https://github.com/user-attachments/assets/37878544-cfda-4a8a-9b42-a7361782c750" 
             poster="https://img.youtube.com/vi/zjGooBuqdSE/maxresdefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="auto" 
             style="object-fit: cover; border-radius: 8px;"></video>
      <br><strong>自動ツール生成</strong><br>必要な動作を伝えるだけで、メタエージェントがツールのコードとスキーマを作成し、そのままワークフローに組み込めます。
    </td>
  </tr>
</table>

> [!NOTE]
> 詳細については、[ドキュメント](https://tencentcloudadp.github.io/youtu-agent/auto_generation/)を参照してください。

## ✨ 特徴

![features](docs/assets/images/header.png)

### デザインコンセプト

- シンプルなデザイン：フレームワークを簡素化し、不必要なコストを避けます。
- モジュール化と設定可能性：柔軟なカスタマイズと新しいコンポーネントの簡単な統合が可能です。
- オープンソースモデルのサポートと低コスト：さまざまなアプリケーションのアクセシビリティとコストパフォーマンスを向上させます。

### コア機能

- openai-agentsに基づいて構築：[openai-agents](https://github.com/openai/openai-agents-python) SDKを基盤としており、ストリーミング、トレーシング、エージェントループの機能を継承しています。これにより、`responses`や`chat.completions` APIとの互換性が保証され、[gpt-oss](https://github.com/openai/gpt-oss)などの多様なモデルにシームレスに対応できます。
- 完全な非同期：高性能かつ効率的な実行を実現し、特に効率的な評価に有利です。
- トレーシングと分析システム：OTELに加えて、`DBTracingProcessor`システムはツールの呼び出しやエージェントのトレースに関する詳細な分析を提供します。（近日リリース予定）

### 自動化

- YAMLベースの設定：構造化され、管理しやすいエージェント設定が可能です。
- 自動エージェント生成：ユーザーのニーズに応じて、エージェント設定を自動的に生成できます。
- ツール生成と最適化：ツールの評価と自動化による最適化が可能で、カスタマイズされたツールの生成機能も将来的にサポートされる予定です。

### 用途例

- 深層/広範な研究：一般的な検索指向のタスクをカバーします。
- ページ生成：特定の入力に基づいてページを生成する例があります。
- トレース収集：トレーニングや研究目的でのデータ収集をサポートします。

## 🤔 なぜYoutu-Agentを選ぶのか？

`Youtu-Agent`は、さまざまなユーザーグループに価値を提供することを目的としています：

### エージェント研究者や大規模言語モデルのトレーナー向け

- 基本的なReActよりも強力で、シンプルながらもパワフルなベースラインとなり、モデルトレーニングやアブレーション研究の優れた出発点となります。
- 実験プロセスを簡素化し、一貫したベンチマークテストを保証するための**ワンクリック評価スクリプト**があります。

### エージェントアプリケーション開発者向け

- 実際のエージェントアプリケーションを構築するための**検証済みで移植可能なフレームワーク**です。
- **使いやすさ**：シンプルなスクリプトと豊富な組み込みツールキットにより、迅速に始めることができます。
- **モジュール設計**：`Environment`や`ContextManager`などの重要なコンポーネントは封装されていますが、高度にカスタマイズ可能です。

### 人工知能やエージェントの愛好家向け

- **実際の使用例**：`/examples`ディレクトリには、ディープラーニングレポートの生成、データ分析、個人ファイルの整理などのタスクが含まれています。
- **シンプルさとデバッグの容易さ**：豊富なツールセットと可視化トレーキングツールにより、開発とデバッグが直感的で簡単になります。

## 🧩 コアコンセプト

- **エージェント（Agent）**：ヒント、ツール、環境が設定された大規模言語モデル。
- **ツールキット（Toolkit）**：エージェントが使用できるツールの集合体。
- **環境（Environment）**：エージェントが操作する世界（例：ブラウザ、シェル）。
- **コンテキストマネージャー（ContextManager）**：エージェントのコンテキストウィンドウを管理するための設定可能なモジュール。
- **ベンチマーク（Benchmark）**：特定のデータセットに対応したワークフローの集合体で、前処理、実行、判断ロジックを含む。

設計および実装の詳細については、[オンラインドキュメント](https://tencentcloudadp.github.io/youtu-agent/)をご覧ください。

## 🚀 すぐに始める

Youtu-Agent は完全なコードとサンプルを提供しており、すぐに使用を開始するのに役立ちます。以下の手順に従って、最初のエージェントを実行するか、[`docker/README.md`](./docker/README.md)を参照してDockerを使用してインタラクティブなウェブページを備えたサンプルを迅速に実行できます。

### 環境準備

リポジトリをクローンして依存関係をインストールします：

> [!NOTE]
> このプロジェクトでは **Python 3.12+** を使用しています。依存関係の管理には [uv](https://github.com/astral-sh/uv) の使用をお勧めします。

まず、環境に Python と uv がインストールされていることを確認し、以下の手順に従ってこのプロジェクトをクローンして依存関係を同期させてください。

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync
source./.venv/bin/activate
cp.env.example.env  # NOTE: 関連する環境変数を設定する必要があります！
```

> [!NOTE]
> `.env` ファイルに LLM API キーなどの関連する環境変数を設定してください。

### すぐに始める

Youtu-Agent には設定ファイルが内蔵されています。例えば、デフォルトの設定ファイル (`configs/agents/simple/base_search.yaml`) では、検索ツールを備えたシンプルなエージェントが定義されています：

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "あなたはウェブを検索できる役立つアシスタントです。"
```

以下のコマンドを使用してインタラクティブな CLI チャットボットを起動できます：

```bash
# NOTE: `.env` に `SERPER_API_KEY` と `JINA_API_KEY` を設定する必要があります（将来的には無料のツールに置き換える予定です）
python scripts/cli_chat.py --config simple/base_search
# 検索ツールを使用しない場合は、以下のコマンドを実行できます
python scripts/cli_chat.py --config simple/base
```

📖 詳細については：[クイックスタートドキュメント](https://tencentcloudadp.github.io/youtu-agent/quickstart) を参照してください。

### サンプルの探索

このリポジトリには直接実行できる複数のサンプルがあります。例えば、特定の研究トピックに基づいて自動的に **SVG インフォグラフィック** を生成することができます：

```bash
python examples/svg_generator/main_web.py
```

> [!NOTE]
> WebUI を使用するには `utu_agent_ui` パッケージをインストールする必要があります。[ドキュメント](https://tencentcloudadp.github.io/youtu-agentfrontend/#installation) を参照してください。

研究トピックを指定すると、エージェントは自動的にネットワーク検索を実行し、関連情報を収集して SVG ビジュアライゼーションを出力します。

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 さらに多くのサンプルについては：[サンプルドキュメント](https://tencentcloudadp.github.io/youtu-agent/examples) を参照してください。

### 評価の実行

Youtu-Agent では標準データセットでのベンチマークテストもサポートしています。例えば、**WebWalkerQA** 上で評価を実行するには：

```bash
# データセットの前処理。このスクリプトは WebWalkerQA データセットをダウンロードして処理し、データベースに保存します。
python scripts/data/process_web_walker_qa.py

# ww.yaml の設定を使用して評価を実行します。迅速な評価のために WebWalkerQA_15 という小さなデータセットを選択しました。
# NOTE: `.env` に `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` を設定する必要があります。`.env.full` を参照してください。
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

結果はローカルに保存され、分析プラットフォームでさらに確認できます。[評価分析](./frontend/exp_analysis/README.md) を参照してください。

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 詳細については：[評価ドキュメント](https://tencentcloudadp.github.io/youtu-agent/eval) を参照してください。

## 📖 さらに詳しく

クイックスタートを終えたら、完全なドキュメントを通じてフレームワークとその機能についてさらに学ぶことができます：

- 📖 **[完全なドキュメント](https://tencentcloudadp.github.io/youtu-agent/)**: コアコンセプト、アーキテクチャ、および高度な機能をご覧ください。
- 🚀 **[クイックスタートガイド](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: 迅速にセットアップして実行するための詳細なガイドです。
- ❓ **[よくある質問](https://tencentcloudadp.github.io/youtu-agent/faq)**: よくある質問とその回答をご覧ください。

## 🙏 お礼の言葉

このプロジェクトは、以下の優れたオープンソースプロジェクトをベースにしています：
- [openai-agents](https://github.com/openai/openai-agents-python)
- [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
- [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 貢献

私たちはコミュニティからの貢献を歓迎します！Youtu-Agentの改善にご協力いただける場合は、まず[**貢献ガイドライン**](./CONTRIBUTING.md)をお読みください。

## 📚 引用

この研究が役立つと思われる場合は、引用を検討してください：

```bibtex
@misc{training_free_grpo,
  title={Training-Free Group Relative Policy Optimization},
  author={Tencent Youtu Lab},
  year={2025},
  eprint={2510.08191},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2510.08191},
}

@misc{youtu-agent-2025,
  title={Youtu-agent: A Simple yet Powerful Agent Framework},
  author={Tencent Youtu Lab},
  year={2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TencentCloudADP/youtu-agent}},
}
```
