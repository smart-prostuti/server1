// index.js
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const crypto = require('node:crypto');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
const port = process.env.PORT || 3001;

app.use(cors({
  origin: [
    'http://localhost:5173',          // Local development
    'https://toolsgovt.netlify.app',  // Netlify frontend
  ],
}));
app.use(express.json());

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

// ---- In-memory recent questions store (per subject|topic) ----
// Map< key, Map<normalizedQuestion, timestamp> >
const recentQuestions = new Map();
const RECENT_TTL_MS = 12 * 60 * 60 * 1000; // 12 hours
const MAX_RECENT_PER_KEY = 250;

const keyFor = (subject, topic) => `${String(subject).trim()}|${String(topic).trim()}`;

// ✅ Fixed regex: removed invalid escapes; placed "-" at end to be literal.
const STRIP_NUMBERING_RE = /^\s*[-—*•]*\d+[.)।:-]?\s*/u;

const normalizeQ = (s) =>
  String(s)
    .replace(STRIP_NUMBERING_RE, '') // strip leading numbering/bullets
    .replace(/\s+/g, ' ')
    .trim();

function purgeExpiredForKey(key) {
  const bucket = recentQuestions.get(key);
  if (!bucket) return;
  const now = Date.now();
  for (const [q, ts] of bucket.entries()) {
    if (now - ts > RECENT_TTL_MS) {
      bucket.delete(q);
    }
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

// Fisher–Yates shuffle for variety
function shuffleInPlace(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ---- Generation configs (tune these for more/less variation) ----
const genCfgQuestions = {
  temperature: 1.15,
  topP: 0.9,
  topK: 40,
  maxOutputTokens: 512,
};

const genCfgQuestionsSpicier = {
  temperature: 1.35,
  topP: 0.95,
  topK: 64,
  maxOutputTokens: 640,
};

const genCfgFeedback = {
  temperature: 0.95,
  topP: 0.9,
  topK: 40,
  maxOutputTokens: 2048,
};

/**
 * --- Generate Questions ---
 * Universal for ICT, Chemistry, Physics, etc.
 */
app.post('/api/generate-questions', async (req, res) => {
  noStore(res);

  const { subject, topic, studentName, count } = req.body;
  if (!subject || !topic || !studentName || !count) {
    console.log('Validation failed: Missing required fields for questions.', req.body);
    return res.status(400).json({ error: 'বিষয়, টপিক, শিক্ষার্থীর নাম এবং প্রশ্নের সংখ্যা উল্লেখ করা আবশ্যক।' });
  }

  try {
    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

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

    // If not enough, retry once with spicier config
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

    // If still more than needed, shuffle then slice to increase perceived variety
    shuffleInPlace(parsed);
    if (parsed.length > count) parsed = parsed.slice(0, count);

    // Track to recent memory
    rememberQuestions(subject, topic, parsed);

    // Number nicely before sending (clients can also parse lines)
    const numbered = parsed.map((q, i) => `${i + 1}. ${q}`).join('\n');
    res.json({ questionsText: numbered });
  } catch (error) {
    console.error('Error calling Gemini API for questions:', error?.message || error);
    res.status(500).json({ error: `প্রশ্ন তৈরি করতে সমস্যা হয়েছে: ${error?.message || 'Unknown error'}` });
  }
});

/**
 * --- Evaluate Multiple Answers ---
 * Universal for ICT, Chemistry, etc.
 */
app.post('/api/submit-multiple-answers', async (req, res) => {
  noStore(res);

  const { studentName, subject, topic, questions, answers } = req.body;

  if (!studentName || !subject || !topic || !Array.isArray(questions) || typeof answers !== 'object' || Object.keys(answers).length === 0) {
    console.log('Validation failed: Missing or invalid fields for multiple answers submission.', req.body);
    return res.status(400).json({ error: 'শিক্ষার্থীর নাম, বিষয়, টপিক, প্রশ্নপত্র এবং উত্তরগুলো আবশ্যক।' });
  }

  try {
    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

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

প্রশ্নপত্র ও উত্তর:
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

// --- Start the server ---
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
