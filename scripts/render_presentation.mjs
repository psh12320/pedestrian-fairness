// Render the saved PPTX for visual review, using the bundled presentation runtime.
import fs from "node:fs/promises";
import path from "node:path";
import {pathToFileURL} from "node:url";

const runtime=process.env.RUNTIME_NODE_MODULES ||
  "C:/Users/Shricharan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
const {FileBlob,PresentationFile}=await import(pathToFileURL(
  path.join(runtime,"@oai/artifact-tool/dist/artifact_tool.mjs")).href);
const source=process.argv[2] || "deliverables/presentation/output/team38-research-talk-v2f-final.pptx";
const output=process.argv[3] || "deliverables/presentation/.build/final-render";
await fs.mkdir(output,{recursive:true});
const deck=await PresentationFile.importPptx(await FileBlob.load(source));
for(const [index,slide] of deck.slides.items.entries()) {
  const png=await deck.export({slide,format:"png",scale:1});
  await fs.writeFile(path.join(output,`slide-${index+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(`Rendered ${deck.slides.items.length} slides to ${output}`);
