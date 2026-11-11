// Deployment settings. Edit `apiBase` to the public URL of the FinRag API (no trailing slash), e.g.
//   apiBase: "https://<user>-finrag.hf.space"
// The API must list this site's origin in api.cors_origins or in FINRAG_CORS_ORIGINS.
window.FINRAG_CONFIG = {
  apiBase: "http://localhost:8000",
  maxQuestionChars: 2000, // keep equal to api.max_question_chars in config.yaml
  requestTimeoutMs: 90000, // a cold start of the free backend can take about a minute
  wakeNoticeMs: 8000, // show "the backend may be waking up" after this long without an answer
  healthPollMs: 5000,
  healthPollMaxMs: 120000,
};
