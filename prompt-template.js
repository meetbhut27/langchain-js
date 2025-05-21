import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// Import environment variables
import * as dotenv from "dotenv";
dotenv.config();

// Instantiate the model
const model = new ChatOpenAI({
  modelName: "meta-llama/llama-3.3-8b-instruct:free",
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
  }, 
  temperature: 0.9,
});

// Create Prompt Template using fromTemplate
// const prompt = ChatPromptTemplate.fromTemplate('Tell a joke about {word}');

// Create Prompt Template from fromMessages
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are a talented chef.  Create a recipe based on a main ingredient provided by the user.",
  ],
  ["human", "{word}"],
]);

const chain = prompt.pipe(model);

const response = await chain.invoke({
  word: "chicken",
});

console.log(response);
