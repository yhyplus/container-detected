async function refreshJob() {
  const root = document.body;
  const jobId = root.dataset.jobId;
  if (!jobId) return;
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {cache: "no-store"});
    if (!response.ok) return;
    const payload = await response.json();
    document.querySelector("#job-status").textContent = payload.job.status;
    document.querySelector("#job-finished").textContent = payload.job.finished_at || "尚未结束";
    document.querySelector("#job-log").textContent = payload.log;
  } catch (_) {
    // Keep the last visible status and retry on the next interval.
  }
}
setInterval(refreshJob, 2000);
