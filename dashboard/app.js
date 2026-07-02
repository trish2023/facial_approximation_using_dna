const stateUrl = "/api/state";
const optionsUrl = "/api/options";
const runButton = document.getElementById("run-button");
const dryRunButton = document.getElementById("dry-run-button");
const refreshButton = document.getElementById("refresh-button");
const vcfSelect = document.getElementById("vcf-select");
const sampleInput = document.getElementById("sample-input");
const outputInput = document.getElementById("output-input");
const skipStage2 = document.getElementById("skip-stage2");
const runStatus = document.getElementById("run-status");
const runMeta = document.getElementById("run-meta");
const subjectId = document.getElementById("subject-id");
const reliability = document.getElementById("reliability");
const finalExists = document.getElementById("final-exists");
const qcExists = document.getElementById("qc-exists");
const overallPill = document.getElementById("overall-pill");
const stagesRoot = document.getElementById("stages");
const logTail = document.getElementById("log-tail");
const previewImage = document.getElementById("preview-image");
const previewLink = document.getElementById("preview-link");
const previewMeta = document.getElementById("preview-meta");
const stageTemplate = document.getElementById("stage-template");

function statusLabel(status) {
  if (status === "complete") return "Complete";
  if (status === "running") return "Running";
  if (status === "failed") return "Failed";
  return "Pending";
}

function statusTone(status) {
  if (status === "complete") return "complete";
  if (status === "running") return "running";
  if (status === "failed") return "failed";
  return "";
}

function formatBytes(sizeKb) {
  if (typeof sizeKb !== "number") return "";
  return `${sizeKb.toFixed(1)} KB`;
}

function renderVerification(items) {
  return items
    .map((item) => {
      const state = item.exists ? "ok" : "missing";
      const extra = item.exists ? `${formatBytes(item.size_kb)}${item.modified ? ` · ${item.modified}` : ""}` : "missing";
      return `<li><span>${item.path}</span><span class="${state}">${item.exists ? "Verified" : "Missing"}${extra ? ` · ${extra}` : ""}</span></li>`;
    })
    .join("");
}

function renderStages(stages) {
  stagesRoot.innerHTML = "";
  stages.forEach((stage) => {
    const node = stageTemplate.content.cloneNode(true);
    const card = node.querySelector(".stage-card");
    node.querySelector("h3").textContent = stage.title;
    node.querySelector("p").textContent = stage.description;
    const badge = node.querySelector(".stage-badge");
    badge.textContent = statusLabel(stage.status);
    badge.dataset.status = statusTone(stage.status);
    node.querySelector(".artifact-list").innerHTML = renderVerification(stage.outputs);
    node.querySelector(".stage-log").textContent = stage.log_excerpt || "No log excerpt captured yet.";
    stagesRoot.appendChild(node);
  });
}

function setRunState(payload) {
  const run = payload.run || {};
  const summary = payload.summary || {};
  const current = run.current_stage || "Idle";
  runStatus.textContent = run.running ? "Pipeline running" : run.error ? "Pipeline failed" : "Idle";
  runMeta.textContent = run.running
    ? `Stage: ${current}`
    : run.error
      ? `Stopped at ${current}.`
      : "No active pipeline.";
  subjectId.textContent = summary.subject_id || run.sample_id || "Unknown";
  reliability.textContent = typeof summary.overall_reliability === "number" ? summary.overall_reliability.toFixed(3) : "-";
  finalExists.textContent = summary.final_composite?.exists ? "Present" : "Missing";
  qcExists.textContent = summary.qc_report?.exists ? "Present" : "Missing";
  overallPill.textContent = run.running ? "Running" : summary.final_composite?.exists ? "Ready" : "Waiting";
  logTail.textContent = payload.log_tail || "No pipeline log yet.";

  if (summary.final_composite?.exists) {
    previewImage.src = `/outputs/final_composite.png?ts=${Date.now()}`;
    previewMeta.textContent = `Final composite saved at ${summary.final_composite.modified || "unknown time"} in ${summary.final_composite.path}.`;
  } else {
    previewImage.removeAttribute("src");
    previewMeta.textContent = "Run the pipeline to generate the final composite preview.";
  }

  renderStages(payload.stages || []);
}

async function loadOptions() {
  const response = await fetch(optionsUrl);
  const data = await response.json();
  const options = data.vcfs || [];
  vcfSelect.innerHTML = options
    .map((item) => {
      const selected = item.path === data.default_vcf ? " selected" : "";
      const suffix = item.sample_id ? ` · sample ${item.sample_id}` : "";
      return `<option value="${item.path}"${selected}>${item.label}${suffix}</option>`;
    })
    .join("");

  const first = options.find((item) => item.path === data.default_vcf) || options[0];
  if (first && first.sample_id) {
    sampleInput.value = first.sample_id;
  }
}

async function refresh() {
  const response = await fetch(stateUrl, { cache: "no-store" });
  const payload = await response.json();
  setRunState(payload);
}

async function runPipeline(dryRun) {
  runButton.disabled = true;
  dryRunButton.disabled = true;
  try {
    const body = {
      vcf: vcfSelect.value,
      sample: sampleInput.value.trim() || null,
      output_dir: outputInput.value.trim() || "outputs",
      skip_stage2: skipStage2.checked,
      dry_run: dryRun,
    };
    await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    await refresh();
    const timer = window.setInterval(async () => {
      await refresh();
      const response = await fetch(stateUrl, { cache: "no-store" });
      const payload = await response.json();
      if (!payload.run?.running) {
        window.clearInterval(timer);
        await refresh();
        runButton.disabled = false;
        dryRunButton.disabled = false;
      }
    }, 1500);
  } catch (error) {
    console.error(error);
    runButton.disabled = false;
    dryRunButton.disabled = false;
    alert("Could not start the pipeline.");
  }
}

runButton.addEventListener("click", () => runPipeline(false));
dryRunButton.addEventListener("click", () => runPipeline(true));
refreshButton.addEventListener("click", refresh);

loadOptions().then(refresh);
window.setInterval(refresh, 4000);
