// index.js
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const crypto = require('node:crypto');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
const port = process.env.PORT || 3001;

app.set('trust proxy', 1);

// ---- CORS ----
// Add any custom domains you use in production here.
const ALLOWED_ORIGINS = [
  'http://localhost:5173',
  'https://toolsgovt.netlify.app',
  // 'https://YOUR-CUSTOM-DOMAIN.com',
];
app.use(cors({ origin: ALLOWED_ORIGINS }));
app.use(express.json({ limit: '1mb' }));

// ---- Health check ----
app.get('/healthz', (_req, res) => res.status(200).send('ok'));

// ---- Gemini init ----
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
  console.error('Error: GEMINI_API_KEY is not set in environment variables.');
  process.exit(1);
}
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

// ---- Utility: response cache control ----
const noStore = (res) => {
  res.set('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
  res.set('Pragma', 'no-cache');
  res.set('Expires', '0');
  res.set('Surrogate-Control', 'no-store');
};

// ---- Helpers for robust JSON from model ----
function parseModelJson(text) {
  if (!text || typeof text !== 'string') {
    throw new Error('Empty model response');
  }

  // Prefer fenced block if present: ```json ... ``` or ``` ... ```
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  let candidate = fenced ? fenced[1] : text;

  // Normalize smart quotes
  candidate = candidate
    .replace(/[\u201C\u201D]/g, '"') // smart double quotes
    .replace(/[\u2018\u2019]/g, "'"); // smart single quotes

  // Extract the first {...} block
  const start = candidate.indexOf('{');
  const end = candidate.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) {
    throw new Error('No JSON object found in model output');
  }
  let jsonSlice = candidate.slice(start, end + 1);

  // Remove trailing commas like ,}
  jsonSlice = jsonSlice.replace(/,\s*([}\]])/g, '$1');

  return JSON.parse(jsonSlice);
}

function coerceAnalysisShape(analysis) {
  if (typeof analysis !== 'object' || analysis === null) analysis = {};
  analysis.summary = typeof analysis.summary === 'string' ? analysis.summary : '';
  analysis.weaknesses = Array.isArray(analysis.weaknesses) ? analysis.weaknesses : [];
  analysis.suggestions = Array.isArray(analysis.suggestions) ? analysis.suggestions : [];
  analysis.encouragement = typeof analysis.encouragement === 'string' ? analysis.encouragement : '';
  return analysis;
}

// ---- In-memory recent questions store (per subject|topic) ----
const recentQuestions = new Map();
const RECENT_TTL_MS = 12 * 60 * 60 * 1000; // 12 hours
const MAX_RECENT_PER_KEY = 250;

const keyFor = (subject, topic) => `${String(subject).trim()}|${String(topic).trim()}`;
// Put "-" at end to be literal.
const STRIP_NUMBERING_RE = /^\s*[-—*•]*\d+[.)।:-]?\s*/u;

const normalizeQ = (s) =>
  String(s)
    .replace(STRIP_NUMBERING_RE, '')
    .replace(/\s+/g, ' ')
    .trim();

function purgeExpiredForKey(key) {
  const bucket = recentQuestions.get(key);
  if (!bucket) return;
  const now = Date.now();
  for (const [q, ts] of bucket.entries()) {
    if (now - ts > RECENT_TTL_MS) bucket.delete(q);
  }
}

function rememberQuestions(subject, topic, qs) {
  const key = keyFor(subject, topic);
  const bucket = recentQuestions.get(key) || new Map();
  const now = Date.now();
  purgeExpiredForKey(key);

  for (const q of qs) {
    const nq = normalizeQ(q);
    if (!nq) continue;
    if (bucket.size >= MAX_RECENT_PER_KEY && !bucket.has(nq)) {
      // drop oldest
      let oldestKey = null;
      let oldestTs = Infinity;
      for (const [qq, ts] of bucket.entries()) {
        if (ts < oldestTs) { oldestTs = ts; oldestKey = qq; }
      }
      if (oldestKey) bucket.delete(oldestKey);
    }
    bucket.set(nq, now);
  }
  recentQuestions.set(key, bucket);
}

function filterOutRecent(subject, topic, rawQuestions) {
  const key = keyFor(subject, topic);
  purgeExpiredForKey(key);
  const bucket = recentQuestions.get(key);
  if (!bucket) return rawQuestions;

  const out = [];
  const seen = new Set();
  for (const line of rawQuestions) {
    const nq = normalizeQ(line);
    if (!nq || seen.has(nq)) continue;
    if (!bucket.has(nq)) {
      out.push(line);
      seen.add(nq);
    }
  }
  return out;
}

function parseQuestionsText(text) {
  return String(text || '')
    .split('\n')
    .map(s => s.trim())
    .filter(Boolean)
    .map(s => s.replace(STRIP_NUMBERING_RE, '').trim())
    .filter((v, i, a) => v && a.indexOf(v) === i);
}

// Fisher–Yates shuffle
function shuffleInPlace(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ---- Generation configs ----
const genCfgQuestions = { temperature: 1.15, topP: 0.9, topK: 40, maxOutputTokens: 512 };
const genCfgQuestionsSpicier = { temperature: 1.35, topP: 0.95, topK: 64, maxOutputTokens: 640 };
const genCfgFeedback = { temperature: 0.95, topP: 0.9, topK: 40, maxOutputTokens: 2048 };

// ---- Models (override via env if needed) ----
const QUESTIONS_MODEL = process.env.GEMINI_QUESTIONS_MODEL || 'gemini-1.5-flash';
const FEEDBACK_MODEL  = process.env.GEMINI_FEEDBACK_MODEL  || 'gemini-1.5-flash';

// =======================================================
// ==============   HSC/Universal Endpoints   ============
// =======================================================

/**
 * Generate Questions
 */
app.post('/api/generate-questions', async (req, res) => {
  noStore(res);

  const { subject, topic, studentName, count } = req.body;
  if (!subject || !topic || !studentName || !count) {
    console.log('Validation failed: Missing required fields for questions.', req.body);
    return res.status(400).json({ error: 'বিষয়, টপিক, শিক্ষার্থীর নাম এবং প্রশ্নের সংখ্যা উল্লেখ করা আবশ্যক।' });
  }

  try {
    const model = genAI.getGenerativeModel({ model: QUESTIONS_MODEL });

    // Nonce to break determinism/caching
    const nonce = `${crypto.randomUUID?.() || Math.random().toString(36).slice(2)}-${Date.now()}`;

    const basePrompt = `
তুমি একজন অভিজ্ঞ ও ভদ্র শিক্ষক।
"${subject}" বিষয়ের "${topic}" টপিক থেকে ${count}টি সংক্ষিপ্ত, স্পষ্ট এবং একে-অপরের থেকে ভিন্ন প্রশ্ন তৈরি করো।
- প্রশ্নগুলো হবে জ্ঞানভিত্তিক, সরাসরি, এবং পুনরাবৃত্তিহীন।
- ছোট/একলাইনি প্রশ্নও রাখতে পারো, তবে অর্থবহ হতে হবে।
- একই ধারণা একাধিকভাবে যেন না আসে।
- প্রতিটি প্রশ্ন নতুন লাইনে সিরিয়াল লিখতে পারো, তবে উত্তর দেবে না।
- ভাষা মানবিক ও সহমর্মী হওয়া চাই।
- প্রশ্নগুলো HSC/কলেজ-লেভেল ধরেই করো।

র‍্যান্ডম টোকেন (ভ্যারিয়েশন নিশ্চিত করতে): ${nonce}
    `.trim();

    // First try
    const first = await model.generateContent({
      contents: [{ role: 'user', parts: [{ text: basePrompt }] }],
      generationConfig: genCfgQuestions,
    });
    const firstText = (await first.response).text();
    let parsed = parseQuestionsText(firstText);

    // Filter out recent repeats
    parsed = filterOutRecent(subject, topic, parsed);

    // Retry once with spicier config if needed
    if (parsed.length < count) {
      const spicy = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: basePrompt + '\n\nভ্যারিয়েশন আরেকটু বাড়াও।' }] }],
        generationConfig: genCfgQuestionsSpicier,
      });
      const spicyText = (await spicy.response).text();
      const parsed2 = filterOutRecent(subject, topic, parseQuestionsText(spicyText));

      // Merge unique
      const merged = [];
      const seen = new Set(parsed.map(normalizeQ));
      for (const q of [...parsed, ...parsed2]) {
        const nq = normalizeQ(q);
        if (!nq || seen.has(nq)) continue;
        merged.push(q);
        seen.add(nq);
      }
      parsed = merged;
    }

    shuffleInPlace(parsed);
    if (parsed.length > count) parsed = parsed.slice(0, count);

    // Track to recent memory
    rememberQuestions(subject, topic, parsed);

    // Number before sending
    const numbered = parsed.map((q, i) => `${i + 1}. ${q}`).join('\n');
    res.json({ questionsText: numbered });
  } catch (error) {
    console.error('Error calling Gemini API for questions:', error?.message || error);
    res.status(500).json({ error: `প্রশ্ন তৈরি করতে সমস্যা হয়েছে: ${error?.message || 'Unknown error'}` });
  }
});

/**
 * Evaluate Multiple Answers (free-form feedback text)
 */
app.post('/api/submit-multiple-answers', async (req, res) => {
  noStore(res);

  const { studentName, subject, topic, questions, answers } = req.body;

  if (!studentName || !subject || !topic || !Array.isArray(questions) || typeof answers !== 'object' || Object.keys(answers).length === 0) {
    console.log('Validation failed: Missing or invalid fields for multiple answers submission.', req.body);
    return res.status(400).json({ error: 'শিক্ষার্থীর নাম, বিষয়, টপিক, প্রশ্নপত্র এবং উত্তরগুলো আবশ্যক।' });
  }

  try {
    const model = genAI.getGenerativeModel({ model: FEEDBACK_MODEL });

    const toneVariants = [
      'বন্ধুসুলভ এবং আত্মবিশ্বাস বাড়ায় এমন টোন',
      'শিক্ষকসুলভ কিন্তু উৎসাহব্যঞ্জক টোন',
      'খুবই নরম ও সহানুভূতিশীল টোন',
      'পরামর্শমূলক ও উদাহরণ-ভিত্তিক টোন',
      'সংক্ষিপ্ত কিন্তু কার্যকরী সুপারিশ–ভিত্তিক টোন'
    ];
    const tone = toneVariants[Math.floor(Math.random() * toneVariants.length)];
    const nonce = `${crypto.randomUUID?.() || Math.random().toString(36).slice(2)}-${Date.now()}`;

    let evaluationPrompt = `
তুমি একজন অভিজ্ঞ ও ভদ্র শিক্ষক।
শিক্ষার্থীর উত্তরগুলো বাংলায় ${tone}-এ মূল্যায়ন করো।
- প্রতিটি প্রশ্নের উত্তর আলাদা করে ছোট ছোট প্যারায় ফিডব্যাক দাও।
- যেখানেই প্রয়োজন, ১–২ লাইনের ছোট উদাহরণ দাও।
- ভুল/আধাভুল হলে কীভাবে ঠিক করা যায় — ২–৩টি বুলেট পয়েন্টে বলো।
- শেষে সামগ্রিক মন্তব্য এবং একটি গ্রেড দাও (চমৎকার/ভালো/উন্নতির প্রয়োজন)।
র‍্যান্ডম টোকেন: ${nonce}

শিক্ষার্থীর নাম: ${studentName}
বিষয়: ${subject}
টপিক: ${topic}

прশ্নপত্র ও উত্তর:
`.trim();

    questions.forEach((question, index) => {
      const studentAnswer = answers[index] || 'কোনো উত্তর দেওয়া হয়নি।';
      evaluationPrompt += `
প্রশ্ন ${index + 1}: ${question}
শিক্ষার্থীর উত্তর: ${studentAnswer}
--------------------
`;
    });

    const result = await model.generateContent({
      contents: [{ role: 'user', parts: [{ text: evaluationPrompt }] }],
      generationConfig: genCfgFeedback,
    });
    const response = await result.response;
    const text = response.text();

    res.json({ feedbackText: text });
  } catch (error) {
    console.error('Error calling Gemini API for multiple answer feedback:', error?.message || error);
    res.status(500).json({ error: `উত্তর জমা দিতে সমস্যা হয়েছে: ${error?.message || 'Unknown error'}` });
  }
});

/**
 * Analyze answers into STRICT JSON (summary/weaknesses/suggestions/encouragement)
 * This matches your Model_Test / SSC frontends that call /api/analyze-answers.
 */
app.post('/api/analyze-answers', async (req, res) => {
  noStore(res);

  const { subject, chapter, totalQuestions, answeredQuestions, wrongAnswers = [] } = req.body || {};
  if (!subject || !chapter || totalQuestions === undefined || answeredQuestions === undefined) {
    return res.status(400).json({ error: 'Subject, chapter, total questions, and answered questions are required.' });
  }

  const wrongAnswersText = wrongAnswers.map((item, index) => {
    const userAns = (item && item.selectedIndex !== null && item.selectedIndex !== undefined)
      ? (item.options?.[item.selectedIndex]?.text ?? 'অজানা অপশন')
      : 'উত্তর দেননি';
    const correctAns = item?.options?.[item.correctIndex]?.text ?? 'অজানা সঠিক উত্তর';
    const exp = item?.explanation || 'কোন ব্যাখ্যা নেই।';

    return `
${index + 1}. প্রশ্ন: ${item?.questionText ?? '—'}
আপনার উত্তর: ${userAns}
সঠিক উত্তর: ${correctAns}
সঠিক উত্তরের ব্যাখ্যা: ${exp}`;
  }).join('\n\n');

  const prompt = `
আপনি একজন অভিজ্ঞ শিক্ষক। একজন শিক্ষার্থীর পরীক্ষার ফলাফলের উপর ভিত্তি করে একটি বিস্তারিত বিশ্লেষণ ও পরামর্শ তৈরি করুন।

পরীক্ষার বিষয়: ${subject}
অধ্যায়: ${chapter}
মোট প্রশ্ন: ${totalQuestions}
উত্তর দেওয়া প্রশ্ন: ${answeredQuestions}
ভুল উত্তর বা উত্তর না দেওয়া প্রশ্ন: ${wrongAnswers.length}

এখানে যেসব প্রশ্নে ভুল হয়েছে বা উত্তর দেয়া হয়নি:
${wrongAnswers.length > 0 ? wrongAnswersText : 'শিক্ষার্থী সব প্রশ্নের সঠিক উত্তর দিয়েছে।'}

শুধুমাত্র নিচের কাঠামো অনুযায়ী **খাঁটি JSON** আউটপুট দিন — কোনো অতিরিক্ত টেক্সট/কোডফেন্স নয়:

{
  "summary": "এই পরীক্ষার সংক্ষিপ্ত সারসংক্ষেপ",
  "weaknesses": [],
  "suggestions": [],
  "encouragement": "শিক্ষার্থীকে উৎসাহিত করার জন্য একটি ছোট বার্তা"
}

যদি কোনো ভুল উত্তর না থাকে, তবে weaknesses, suggestions অ্যারে খালি রাখবেন।
`.trim();

  try {
    const model = genAI.getGenerativeModel({ model: FEEDBACK_MODEL });

    // Ask for application/json; if SDK ignores, we'll still parse robustly.
    const result = await model.generateContent({
      contents: [{ role: 'user', parts: [{ text: prompt }]}],
      generationConfig: { ...genCfgFeedback, responseMimeType: 'application/json' },
    });

    const rawText = result?.response?.text?.() ?? '';
    // Log once to debug server issues (remove if too noisy)
    console.log('analyze-answers raw:', rawText.slice(0, 400) + (rawText.length > 400 ? ' ...' : ''));

    let analysis;
    try {
      analysis = parseModelJson(rawText.trim());
    } catch (parseErr) {
      console.error('JSON parse failed. Raw model output:\n', rawText);
      return res.status(500).json({ error: 'Invalid JSON from Gemini', raw: rawText });
    }

    analysis = coerceAnalysisShape(analysis);
    return res.json({ analysis });
  } catch (error) {
    console.error('Error calling Gemini API for analysis:', error?.message || error);
    return res.status(500).json({ error: `বিশ্লেষণ তৈরি করতে সমস্যা হয়েছে: ${error?.message || 'Unknown error'}` });
  }
});

// --- Fallback 404 (helps spot wrong paths) ---
app.use((req, res) => {
  res.status(404).json({ error: `Route not found: ${req.method} ${req.originalUrl}` });
});

// --- Start the server ---
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
