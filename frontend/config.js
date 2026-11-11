// Settings. `apiBase` is empty (= same origin) because the API serves this UI itself: open http://localhost:8000.
// To host the UI somewhere else, set it to the API's URL (no trailing slash) and list the UI's origin in
// api.cors_origins or FINRAG_CORS_ORIGINS, and allow the API host in the Content-Security-Policy (connect-src).
window.FINRAG_CONFIG = {
  apiBase: "",
  maxQuestionChars: 2000, // keep equal to api.max_question_chars in config.yaml
  requestTimeoutMs: 90000, // a cold start of the free backend can take about a minute
  wakeNoticeMs: 8000, // show "the backend may be waking up" after this long without an answer
  healthPollMs: 5000,
  healthPollMaxMs: 120000,
};
