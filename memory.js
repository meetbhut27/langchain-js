import * as dotenv from "dotenv";
dotenv.config();

import readline from "readline";

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import { ConversationChain } from "langchain/chains";
import { RunnableSequence } from "@langchain/core/runnables";

// Memory
import { BufferMemory } from "langchain/memory";
import { UpstashRedisChatMessageHistory } from "@langchain/community/stores/message/upstash_redis";

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const model = new ChatOpenAI({
  modelName: "meta-llama/llama-3.3-8b-instruct:free",
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
  },
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(
  `You are an AI assistant called Max. You are here to help answer questions and provide information to the best of your ability.
  Chat History: {history}
  {input}`
);

const upstashMessageHistory = new UpstashRedisChatMessageHistory({
  sessionId: "lagchain-chat",
  config: {
    url: process.env.UPSTASH_REDIS_URL,
    token: process.env.UPSTASH_REST_TOKEN,
  },
});

const memory = new BufferMemory({
  memoryKey: "history",
  chatHistory: upstashMessageHistory,
});

// Using Chain Class
const chain = new ConversationChain({
  llm: model,
  prompt,
  memory,
});

function askQuestion() {
  
  rl.question("User: ", async (input) => {
    
    if (input.toLowerCase() === "exit") {
      rl.close();
      return;
    }
    
    const response = await chain.invoke({
      input: input,
    });
    
    console.log("Output: ", response.response);
    
    askQuestion();
  });
}

askQuestion();
// console.log("Chat Memory", await memory.loadMemoryVariables());
