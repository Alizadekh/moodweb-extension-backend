import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

const ALLOWED_MOODS = [
  "Sad",
  "Happy",
  "Excited",
  "Motivated",
  "Stressed",
  "Angry",
];

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

const openai = new OpenAI({
  apiKey: process.env.DASHSCOPE_API_KEY,
  baseURL: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
});

async function detectLanguage(text) {
  try {
    const completion = await openai.chat.completions.create({
      model: "qwen-plus",
      messages: [
        {
          role: "system",
          content:
            "You are a language detector. Return only the language code (e.g., 'en', 'tr', 'es', 'fr', etc.) without any additional text or explanation.",
        },
        {
          role: "user",
          content: `Detect the language of this text: ${text}`,
        },
      ],
      temperature: 0,
      max_tokens: 2,
    });

    if (
      !completion.choices ||
      completion.choices.length === 0 ||
      !completion.choices[0].message ||
      !completion.choices[0].message.content
    ) {
      console.error(
        "Invalid response from OpenAI (detectLanguage):",
        completion
      );
      throw new Error(
        "Invalid response from OpenAI API for language detection"
      );
    }

    const language = completion.choices[0].message.content.trim().toLowerCase();
    if (!/^[a-z]{2}$/.test(language)) {
      throw new Error("Invalid language code format");
    }
    return language;
  } catch (error) {
    console.error("Error in detectLanguage:", error);
    throw error instanceof Error
      ? error
      : new Error("Language detection failed");
  }
}

async function getMoodBasedQuote(mood, language) {
  try {
    const completion = await openai.chat.completions.create({
      model: "qwen-plus",
      messages: [
        {
          role: "system",
          content: `You are a helpful assistant that provides inspiring quotes in the specified language. Provide a short, single inspiring quote that matches the given mood. Always respond in the requested language, without any additional text or explanation.`,
        },
        {
          role: "user",
          content: `Generate one inspiring quote in ${language} that would be suitable for someone feeling ${mood.toLowerCase()}. The quote should be in ${language} only.`,
        },
      ],
      temperature: 0.7,
      max_tokens: 100,
    });

    if (
      !completion.choices ||
      completion.choices.length === 0 ||
      !completion.choices[0].message ||
      !completion.choices[0].message.content
    ) {
      console.error(
        "Invalid response from OpenAI (getMoodBasedQuote):",
        completion
      );
      throw new Error("Invalid response from OpenAI API for quote generation");
    }

    return completion.choices[0].message.content.trim();
  } catch (error) {
    console.error("Error in getMoodBasedQuote:", error);
    throw error instanceof Error ? error : new Error("Quote generation failed");
  }
}

async function analyzeEmotion(text) {
  try {
    const completion = await openai.chat.completions.create({
      model: "qwen-plus",
      messages: [
        {
          role: "system",
          content: `Perform single-word emotion analysis. Possible responses are only: ${ALLOWED_MOODS.join(
            ", "
          )}. Return exactly one word from the list, without any additional text or explanation.`,
        },
        {
          role: "user",
          content: `Analyze the emotion in this text: ${text}`,
        },
      ],
      temperature: 0.7,
      max_tokens: 4,
    });

    if (
      !completion.choices ||
      completion.choices.length === 0 ||
      !completion.choices[0].message ||
      !completion.choices[0].message.content
    ) {
      console.error(
        "Invalid response from OpenAI (analyzeEmotion):",
        completion
      );
      throw new Error("Invalid response from OpenAI API for emotion analysis");
    }

    const result = completion.choices[0].message.content.trim();

    if (!ALLOWED_MOODS.includes(result)) {
      console.error("Invalid mood returned from OpenAI:", result);
      throw new Error(`API returned invalid emotion: ${result}`);
    }

    return result;
  } catch (error) {
    console.error("Error in analyzeEmotion:", error);
    throw error instanceof Error ? error : new Error("Emotion analysis failed");
  }
}

async function getMoodBasedMediaRecommendation(mood, language, userInput) {
  try {
    const completion = await openai.chat.completions.create({
      model: "qwen-plus",
      messages: [
        {
          role: "system",
          content: `You are a music recommendation assistant. Provide a short, natural, and fluent sentence recommending a specific song that suits the given mood and context (based on user input), including the artist and song name (e.g., "Maybe listen to Emin Sabitoğlu - Sakit Gecə, it might help you relax" or "Bəlkə Emin Sabitoğlu - Sakit Gecə dinləsən, bir az rahatlaya bilərsən"). Avoid irrelevant suggestions like love songs for sadness caused by failure. Use proper grammar and phrasing in the language specified by ${language}. Respond with only the sentence.`,
        },
        {
          role: "user",
          content: `Recommend a song in ${language} for someone feeling ${mood.toLowerCase()} based on this context: "${userInput}".`,
        },
      ],
      temperature: 0.7,
      max_tokens: 50,
    });

    if (
      !completion.choices ||
      completion.choices.length === 0 ||
      !completion.choices[0].message ||
      !completion.choices[0].message.content
    ) {
      console.error(
        "Invalid response from OpenAI (getMoodBasedMediaRecommendation):",
        completion
      );
      throw new Error(
        "Invalid response from OpenAI API for media recommendation"
      );
    }

    return completion.choices[0].message.content.trim();
  } catch (error) {
    console.error("Error in getMoodBasedMediaRecommendation:", error);
    throw error instanceof Error
      ? error
      : new Error("Media recommendation failed");
  }
}

async function getMoodBasedActivityRecommendation(mood, language, userInput) {
  try {
    const completion = await openai.chat.completions.create({
      model: "qwen-plus",
      messages: [
        {
          role: "system",
          content: `You are a supportive wellness assistant. Provide a short, practical, and motivating activity recommendation as a full sentence that suits the given mood and context (based on user input). If the mood is negative (e.g., Sad, Stressed), focus on problem-solving and encouragement (e.g., "You seem sad, maybe planning a new study strategy could lift your spirits" or "Kədərli görünürsən, bəlkə imtahan üçün yeni strategiya düşünsən, bu səni ruhlandırar"). If the mood is positive (e.g., Happy, Excited), boost their motivation further with a goal-oriented suggestion (e.g., "You seem happy, maybe channel this energy into starting a new project" or "Sevincli görünürsən, bəlkə bu enerjini yeni bir layihəyə başlamaq üçün istifadə etsən"). Use proper grammar and natural phrasing in the language specified by ${language}. Mention the mood explicitly. Respond with only the sentence.`,
        },
        {
          role: "user",
          content: `Recommend an activity in ${language} for someone feeling ${mood.toLowerCase()} based on this context: "${userInput}".`,
        },
      ],
      temperature: 0.7,
      max_tokens: 80,
    });

    if (
      !completion.choices ||
      completion.choices.length === 0 ||
      !completion.choices[0].message ||
      !completion.choices[0].message.content
    ) {
      console.error(
        "Invalid response from OpenAI (getMoodBasedActivityRecommendation):",
        completion
      );
      throw new Error(
        "Invalid response from OpenAI API for activity recommendation"
      );
    }

    return completion.choices[0].message.content.trim();
  } catch (error) {
    console.error("Error in getMoodBasedActivityRecommendation:", error);
    throw error instanceof Error
      ? error
      : new Error("Activity recommendation failed");
  }
}

export default async function handler(req, res) {
  if (req.method === "OPTIONS") {
    res.setHeader(
      "Access-Control-Allow-Origin",
      corsHeaders["Access-Control-Allow-Origin"]
    );
    res.setHeader(
      "Access-Control-Allow-Methods",
      corsHeaders["Access-Control-Allow-Methods"]
    );
    res.setHeader(
      "Access-Control-Allow-Headers",
      corsHeaders["Access-Control-Allow-Headers"]
    );
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    res.setHeader(
      "Access-Control-Allow-Origin",
      corsHeaders["Access-Control-Allow-Origin"]
    );
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const { userInput } = req.body;

    if (
      !userInput ||
      typeof userInput !== "string" ||
      userInput.trim() === ""
    ) {
      res.setHeader(
        "Access-Control-Allow-Origin",
        corsHeaders["Access-Control-Allow-Origin"]
      );
      return res
        .status(400)
        .json({ error: "Invalid input. Please provide a text string." });
    }

    const language = await detectLanguage(userInput);
    const mood = await analyzeEmotion(userInput);
    const quote = await getMoodBasedQuote(mood, language);
    const mediaRecommendation = await getMoodBasedMediaRecommendation(
      mood,
      language,
      userInput
    );
    const activityRecommendation = await getMoodBasedActivityRecommendation(
      mood,
      language,
      userInput
    );

    res.setHeader(
      "Access-Control-Allow-Origin",
      corsHeaders["Access-Control-Allow-Origin"]
    );
    res.status(200).json({
      mood,
      quote,
      language,
      mediaRecommendation,
      activityRecommendation,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error in /api/analyze-mood:", error);
    res.setHeader(
      "Access-Control-Allow-Origin",
      corsHeaders["Access-Control-Allow-Origin"]
    );
    res.status(500).json({
      error: "An internal server error occurred.",
      details:
        process.env.NODE_ENV === "development" ? error.message : undefined,
    });
  }
}

console.log("To test locall: http://localhost:3000/api/analyze-mood");
