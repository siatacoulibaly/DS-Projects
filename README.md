# DS-Projects

## Data Science Project Portfolio
Welcome! This repository is a showcase of my personal projects, highlighting my journey and passion for data science. It demonstrates my skills in a variety of tools, languages, and techniques. Each project listed below includes a brief description of its purpose, the technologies used, and the data sources.

## A2A Trip Planner: An AI-Powered Itinerary Agent 🗓️
This project features an AI agent designed to automate trip planning. The agent retrieves a user-specified file from Google Drive, extracts key information, and generates a detailed itinerary for a given date range. After drafting the plan, it sends a summary via email for user approval. If the user declines, the workflow initiates an iterative process to refine the plan based on their feedback. Upon approval, the final itinerary is saved as a Google Sheet in their Drive. This workflow was built using n8n and the OpenAI chat model.

## Price-Return Forecasting with ARIMA 📈
This project explores historical stock market data, focusing on Amazon's (AMZN) log returns over the past five years. Using Python, I analyze and model the data with an ARIMA (Autoregressive Integrated Moving Average) model to forecast future returns. The notebook includes all the necessary steps for data retrieval, preprocessing, and model implementation. A subsequent update will include the calculations to convert the forecasted returns back to price values.

## RAG Financial Data Analysis Assistant 🤖
This project implements a Retrieval-Augmented Generation (RAG) system using Python and the OpenAI chat model. The system leverages Optical Character Recognition (OCR) to process and extract text from financial data analysis course PDFs. By grounding the chat model's responses in this specific, unstructured data, the agent can provide accurate and contextually relevant answers to user queries, effectively acting as a specialized financial data assistant. This demonstrates a practical application of RAG for creating domain-specific knowledge chatbots.
