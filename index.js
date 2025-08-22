// Server/index.js (এই ফাইলটি আপনার Server GitHub রিপোজিটরির রুটে থাকবে)

require('dotenv').config(); // লোকাল ডেভেলপমেন্টের জন্য
const express = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const cors = require('cors');

const app = express();
const port = process.env.PORT || 3001; // লোকাল ডেভেলপমেন্টের জন্য

// CORS মিডলওয়্যার সেটআপ
// এখানে আপনার Netlify ফ্রন্টএন্ডের URL টি যোগ করতে হবে
// এবং Render থেকে deploy হওয়া ব্যাকএন্ডের URL (যদি আপনি নিজে deploy করেন)
app.use(cors({
  origin: [
    'http://localhost:5173', // লোকাল ডেভেলপমেন্টের জন্য
    'https://toolsgovt.netlify.app', // <-- আপনার Netlify ফ্রন্টএন্ড URL
  ]
}));
app.use(express.json());

// API Key এনভায়রনমেন্ট ভেরিয়েবল থেকে পাবে (লোকালে .env থেকে, Render-এ Render এর কনফিগারেশন থেকে)
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
  console.error("Error: GEMINI_API_KEY is not set in environment variables.");
  // প্রোডাকশন এনভায়রনমেন্টে API Key না থাকলে সার্ভার ত্রুটি দেবে বা বন্ধ হবে
  process.exit(1); // এই লাইনটি প্রোডাকশনের জন্য ভালো, লোকাল ডেভের জন্য কমেন্ট করে রাখতে পারেন
}

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

app.post('/api/generate-application', async (req, res) => {
  // ফ্রন্টএন্ড থেকে পাঠানো ডেটা রিসিভ করা হচ্ছে
  const { governmentOfficeName, applicantName, topic, additionalPrompt } = req.body;

  // ইনপুট ভ্যালিডেশন
  if (!governmentOfficeName || !applicantName || !topic) {
    console.log('Validation failed: Missing required fields in request body.', req.body);
    return res.status(400).json({ error: 'সরকারি অফিসের নাম, আপনার নাম এবং বিষয় উল্লেখ করা আবশ্যক।' });
  }

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

    // বাংলাদেশ সময় (ঢাকা) অনুযায়ী বর্তমান তারিখ ও সময়
    const bangladeshTimeZone = 'Asia/Dhaka';
    const optionsDate = { day: 'numeric', month: 'long', year: 'numeric', timeZone: bangladeshTimeZone };
    const optionsTime = { hour: 'numeric', minute: 'numeric', second: 'numeric', hour12: true, timeZone: bangladeshTimeZone };
    const currentDate = new Date().toLocaleDateString('bn-BD', optionsDate);
    const currentTime = new Date().toLocaleTimeString('bn-BD', optionsTime);

    // প্রম্পট তৈরি করা হচ্ছে
    const prompt = `
      বর্তমান তারিখ ও সময় (বাংলাদেশ): ${currentDate}, ${currentTime}।
      বাংলায় একটি অফিসিয়াল সরকারি আবেদন পত্র লিখুন।

      আবেদনপত্রের প্রাপক: ${governmentOfficeName}
      আবেদনকারীর নাম: ${applicantName}
      আবেদনের বিষয়: ${topic}
      ${additionalPrompt ? `আবেদনপত্রের বডিতে এই অতিরিক্ত তথ্যগুলো যোগ করুন: "${additionalPrompt}"` : ''}

      আবেদনপত্রের কাঠামো নিম্নরূপ হবে এবং বাংলাদেশের সরকারি অফিসিয়াল আবেদনের রীতি কঠোরভাবে অনুসরণ করবে:

      তারিখ: ${currentDate}

      বরাবর,
      ${governmentOfficeName}
      [এখানে অফিসের একটি জেনেরিক ঠিকানা যোগ করুন, যেমন: সিলেট, বাংলাদেশ]

      বিষয়: ${topic}

      জনাব,

      সবিনয় নিবেদন এই যে, [এখানে "${topic}" এর উপর ভিত্তি করে আবেদনপত্রের মূল বডি লিখুন। "${additionalPrompt}" যদি দেওয়া থাকে, তবে সেটি প্রাসঙ্গিকভাবে বডিতে অন্তর্ভুক্ত করুন। আবেদনপত্রের উদ্দেশ্য স্পষ্ট করুন এবং সংক্ষিপ্ত রাখুন। বাংলাদেশের অফিসিয়াল ভাষা ও বিনয় ব্যবহার করুন]।

      অতএব, জনাবের নিকট বিনীত প্রার্থনা এই যে, [এখানে "${topic}" এর উপর ভিত্তি করে সুনির্দিষ্ট অনুরোধটি স্পষ্টভাবে উল্লেখ করুন এবং প্রয়োজনীয় পদক্ষেপের জন্য প্রার্থনা করুন]।

      বিনীত নিবেদক,
      ${applicantName}
      [আপনার স্বাক্ষর]
      [আপনার পদবি, যদি প্রযোজ্য হয়]
      [আপনার ফোন নম্বর, যদি প্রযোজ্য হয়]
      [আপনার ইমেইল, যদি প্রযোজ্য হয়]

      অনুগ্রহ করে শুধুমাত্র আবেদন পত্রের সম্পূর্ণ টেক্সট দিন, অন্য কোন অতিরিক্ত কথা নয়। ফরম্যাটটি কঠোরভাবে অনুসরণ করুন।
    `;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();

    res.json({ applicationText: text });

  } catch (error) {
    console.error('Error calling Gemini API:', error.message);
    res.status(500).json({ error: `অ্যাপ্লিকেশন তৈরি করতে সমস্যা হয়েছে: ${error.message}` });
  }
});

// লোকাল ডেভেলপমেন্টের জন্য সার্ভার চালু করে
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});