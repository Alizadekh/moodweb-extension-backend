import express from "express";
import OpenAI from "openai";
import cors from "cors";
import dotenv from "dotenv";
import morgan from "morgan";

dotenv.config();

const ALLOWED_MOODS = [
  "Sad",
  "Happy",
  "Excited",
  "Motivated",
  "Stressed",
  "Angry",
];
const PORT = process.env.PORT || 3000;

const app = express();

const openai = new OpenAI({
  apiKey: process.env.DASHSCOPE_API_KEY,
  baseURL: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
});

app.use(express.json());
app.use(
  cors({
    origin: "http://localhost:3001",
    methods: "POST",
    allowedHeaders: "Content-Type",
  })
);
app.use(morgan("dev"));

async function detectLanguage(text) {
  try {
    const completion = await openai.chat.completions.create({
      model: "qwen-plus",
      messages: [
        {
          role: "system",
          content:
            "You are a language detector. Return only the language code (e.g., 'en', 'tr', 'es', 'fr', etc.)",
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
      throw new Error("Invalid response from OpenAI API");
    }

    return completion.choices[0].message.content.trim();
  } catch (error) {
    console.error("Error in detectLanguage:", error);
    throw error;
  }
}

async function getMoodBasedQuote(mood, language) {
  try {
    const completion = await openai.chat.completions.create({
      model: "qwen-plus",
      messages: [
        {
          role: "system",
          content: `You are a helpful assistant that provides inspiring quotes in the specified language. Provide a quote that matches the given mood. Always respond in the requested language.`,
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
      throw new Error("Invalid response from OpenAI API");
    }

    return completion.choices[0].message.content.trim();
  } catch (error) {
    console.error("Error in getMoodBasedQuote:", error);
    throw error;
  }
}

async function analyzeEmotion(text) {
  try {
    const completion = await openai.chat.completions.create({
      model: "qwen-plus",
      messages: [
        {
          role: "system",
          content: `Perform single-word emotion analysis. Possible responses: ${ALLOWED_MOODS.join(
            ", "
          )}.`,
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
      throw new Error("Invalid response from OpenAI API");
    }

    const result = completion.choices[0].message.content.trim();

    if (!ALLOWED_MOODS.includes(result)) {
      console.error("Invalid mood returned from OpenAI:", result);
      throw new Error("API returned an invalid emotion");
    }

    return result;
  } catch (error) {
    console.error("Error in analyzeEmotion:", error);
    throw error;
  }
}

app.post("/analyze-mood", async (req, res) => {
  try {
    const { userInput } = req.body;

    if (
      !userInput ||
      typeof userInput !== "string" ||
      userInput.trim() === ""
    ) {
      return res
        .status(400)
        .json({ error: "Invalid input. Please provide a text string." });
    }

    const language = await detectLanguage(userInput);
    const mood = await analyzeEmotion(userInput);
    const quote = await getMoodBasedQuote(mood, language);

    res.json({ mood, quote, language });
  } catch (error) {
    console.error("Error in /analyze-mood:", error);

    res.status(500).json({
      error: "An internal server error occurred.",
      details:
        process.env.NODE_ENV === "development" ? error.message : undefined,
    });
  }
});

app.use((req, res) => {
  res.status(404).json({ error: "Endpoint not found" });
});

app.use((err, req, res, next) => {
  console.error("Global error:", err);
  res.status(500).json({ error: "Server error" });
});

app.listen(PORT, () => {
  console.log(`Server is running at: http://localhost:${PORT}`);
});
