import Link from "next/link";

import { EvalResults } from "@/components/eval-results";
import { Figure, Measure, MetaPairs } from "@/components/essay";
import { FlowDiagram } from "@/components/flow-diagram";
import { HeroDemo } from "@/components/hero-demo";
import { MixtureTable } from "@/components/mixture";
import { StackTabs } from "@/components/stack";

export default function OverviewPage() {
  return (
    <article>
      <Measure className="essay">
        <h1 className="essay-title">
          Training a detector to read AI image fingerprints
        </h1>
        <p className="essay-kicker mt-3">
          TikTok TechJam 2026 · Track 5 · DINOv3 ViT-L
        </p>
        <p className="mt-6">
          The budget is two billion parameters. We used 302 million: DINOv3 ViT-L,
          fully fine-tuned, with a global head for the page-level verdict and a
          local head for a 32×32 patch map. One forward pass returns both.
        </p>
        <p>
          Training is public data only, 2.58 million usable images, weighted by
          how hard each source is, then run through wild-simulation augmentation
          so JPEG and resize do not wipe the fingerprint. The rest of this page
          is the architecture, the held-out numbers, the mixture, and the tools.
          {" "}
          <Link href="/analyze">Analyze</Link> runs the checkpoint on an image
          you bring.
        </p>
      </Measure>

      <div className="mt-6">
        <HeroDemo />
      </div>

      <Measure className="mt-10">
        <MetaPairs
          rows={[
            {
              label: "Backbone",
              value: "DINOv3 ViT-L/16",
              detail: "full continuation fine-tuning",
            },
            {
              label: "Parameters",
              value: "302M",
              detail: "15% of the 2B budget",
            },
            {
              label: "Input",
              value: "512 × 512",
              detail: "32 × 32 patch grid",
            },
            {
              label: "Output",
              value: "Verdict + heatmap",
              detail: "global head + per-patch local head",
            },
          ]}
        />
      </Measure>

      <Measure className="essay mt-10">
        <h2 className="essay-title">How a verdict is made</h2>
        <p className="mt-4">
          Seer is one network, not a classifier plus a separate localizer. A
          DINOv3 ViT-L/16 is fine-tuned end to end so the features themselves
          learn generator traces.
        </p>
        <p>
          We put a heatmap on every result so the verdict is not a black box:
          it shows which patches the model treats as generated.
        </p>
      </Measure>
      <div className="mt-6">
        <Figure
          wash
          caption="Input image, DINOv3 ViT-L/16, then a fork into the global head and the 32×32 local head."
        >
          <FlowDiagram />
        </Figure>
      </div>

      <Measure className="essay mt-10">
        <h2 className="essay-title">Held-out numbers</h2>
        <p className="mt-4">
          These are Seer&apos;s own held-out scores from various community
          datasets. The NTIRE 2026 open-test leaderboard sits under the table
          so the public-test AUROC can be read next to the published entries.
        </p>
      </Measure>
      <div className="mt-6">
        <EvalResults />
      </div>

      <Measure className="essay mt-10">
        <h2 className="essay-title">The mixture</h2>
        <p className="mt-4">
          Ten public sources, weighted by measured difficulty. 2.58 million
          usable images: 1.70 million fake and 875 thousand real. I open a
          source for what is in it and how to fetch it. Weights decide draw
          probability, not disk usage.
        </p>
      </Measure>
      <div className="mt-6">
        <MixtureTable />
      </div>

      <Measure className="essay mt-10">
        <h2 className="essay-title">What I trained with</h2>
        <p className="mt-4">
          The tools, models, libraries, and data that produced the numbers
          above. Only the parts that shaped the result.
        </p>
      </Measure>
      <div className="mt-6">
        <StackTabs />
      </div>
    </article>
  );
}
