from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import numpy as np
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio
from langchain_qdrant import Qdrant


load_dotenv()
headers = {"User-Agent": os.getenv("USER_AGENT", "BraedenBot/1.0")}

class BraedenBot:
    def __init__(self):
        print("Initializing BraedenBot...")
        self.client = OpenAI()
        self.client2 =  OpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        self.system_prompt = """
            You are BraedenBot, a helpful assistant trained only on Braeden documentation.

            If user send's general conversation message example Hi/Hello respond quickly as {"step":"output","content":"Hello from Braeden Bot!!"}

            You work in a START → PLAN → ANALYZE → RETRIEVE → SYNTHESIZE → OUTPUT workflow when answering user queries.

            If the answer requires referencing a specific page, provide the exact URL from the context.
            
            If no relevant information is found, say: "I couldn't find relevant information about that in the braeden docs"

            When answering:
            - Use the exact content from the documentation rather than summarizing or paraphrasing it, especially for code examples, step-by-step instructions, and technical details.
            - Always strive to provide clear, complete, and detailed explanations.
            - Do not skip steps in your thinking or your output — be explicit and provide thorough reasoning at each stage.
            - If multiple parts of the documentation are relevant, combine them carefully and maintain all context and structure.
            - If the answer requires referencing a specific page, provide the exact URL from the documentation context.
            - If the code is available than add the code also.
            - If no relevant information is found, say: "I couldn't find relevant information about that in the braeden docs."

            1. PLAN:
            - Analyze the user's query carefully.
            - Break down complex questions into simpler components.
            - Identify key concepts and terms that need to be addressed.
            - Break down query in a step back prompting and chain of thought process.

            2. ANALYZE:
            - Look through the retrieved context.
            - Identify the most relevant pieces of information.
            - Find other relevant information that might be related to the query.
            - Consider how different documents might relate to each other and if required use multiple documents to answer the query.

            3. RETRIEVE:
            - Identify and extract the exact content from documentation that answers the query.
            - When code examples exist in the documentation, include them exactly as they appear.
            - Extract all URLs that might be useful for citations.
            - Preserve the original structure and formatting of the documentation wherever possible.

            4. SYNTHESIZE:
            - Use the exact content from the documentation as much as possible.
            - Maintain the original organization, headings, and structure from the documentation.
            - Only synthesize information if multiple documents need to be combined.
            - Do NOT rewrite or paraphrase documentation content unless absolutely necessary.

            5. OUTPUT:
            - Reproduce the exact content from the documentation as your primary response.
            - Keep the original section headings, code formatting, and examples intact.
            - If content spans multiple documents, clearly indicate where each part comes from.
            - Always include source URLs.

            RULES:
            - Base your answers only on the Braeden documentation context provided.
            - Reproduce the exact content from the documentation whenever possible, especially code examples.
            - Never guess or make up information. If uncertain, say the answer is not found.
            - Preserve original formatting, code blocks, and examples exactly as they appear in the documentation.
            - Prioritize verbatim content from the documentation over your own explanations.
            - Always follow JSON format for output.

            IMPORTANT: You must respond using only the following JSON format:

            {
            "step": "<one of: plan, analyze, retrieve, synthesize, output>",
            "content": "<your response content here>"
            }

            Never include anything outside this JSON. No explanations, no extra formatting, no markdown.

            Example 1:
            User query: "What is Braeden email id?"

            Output:
            {
                "step": "plan",
                "content": "User wants to know about Braeden email id. I'll find documentation about Braeden email id."
            }

            Output:
            {
                "step": "analyze",
                "content": "I found documentation about Braeden email id."
            }

            Output:
            {
                "step": "retrieve",
                "content": "The documentation explicitly covers Braeden email id in detail here: - https://braeden.com/" 
            }

            Output:
            {
                "step": "synthesize",
                "content": "I'll extract the exact content from the documentation about Braeden email id, preserving all examples, headings, and formatting."
            }

            Output:
            {
                "step": "output",
                "content": {{Exact content from the documentation}}
            }
        """
        
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        self.context = ""
        
        print("Setting up vector store and retriever...")
        self.retriever = self.setup_retriever()
        print("Braeden Bot initialization completed.")
    
    def load_sitemap(self):
        print("Loading sitemap...")
        sitemap_loader = SitemapLoader(web_path="sitemap.xml", is_local=True)
        docs = sitemap_loader.load()
        
        urls = [doc.metadata["source"] for doc in docs]
        page_content = [doc.page_content for doc in docs]
        
        print(f"Loaded {len(docs)} documents from sitemap")
        return docs
    
    def split_text(self, data):
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        for doc in data:
            source = doc.metadata.get("source", "No source found")
            
            # Extract a title from the URL
            title_segment = source.strip("/").split("/")[-1]
            title = title_segment.replace("-", " ").title()

            doc.page_content = f"{doc.page_content}\n\n[{title}]({source})"
            
        texts = text_splitter.split_documents(documents=data)
        print(f"Split into {len(texts)} chunks")
        return texts
    
    def reciprocal_rank_fusion(self, results_list, k=60):
        scores = {}
        for result_set in results_list:
            for rank, doc in enumerate(result_set[:k]):
                doc_id = doc.metadata.get("source", str(id(doc)))
                score = 1 / (60 + rank)
                scores[doc_id] = scores.get(doc_id, 0) + score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        doc_map = {doc.metadata.get("source", str(id(doc))): doc for result in results_list for doc in result}
        fused_docs = [doc_map[doc_id] for doc_id, _ in sorted_docs]
        return fused_docs
    

    # async def generate_related_queries(self, user_query):
    #     system_prompt = """Generate three helpful, semantically diverse variations of the user's question 
    #             to improve retrieval in a documentation search engine. Return them as a JSON list of strings.
    #             If the given query is direct question then don't generate multiple queries.
    #             example for direct question: where is braeden located. This is one of the direct question
    #             example for indirect questions: who are braeden clients or what are braeden services"""

    #     messages = [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_query}
    #     ]

    #     try:
    #         # Run sync call in a thread-safe async wrapper
    #         response = await asyncio.to_thread(
    #             self.client2.chat.completions.create,
    #             model="gemini-2.0-flash",
    #             messages=messages,
    #             response_format={"type": "json_object"}
    #         )
    #         content = response.choices[0].message.content
    #         queries = json.loads(content)
    #         print("These are the queries:", queries)
    #         if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
    #             return queries
    #     except Exception as e:
    #         print(f"❌ Failed to generate related queries with AI: {e}")

    #     return [
    #         user_query,
    #         f"Explain: {user_query}",
    #         f"Step-by-step answer: {user_query}"
    #     ]
 
    def get_context_for_query(self, query):
        print(f"🔍 Generating related queries for: {query}")
        
        # queries = await self.generate_related_queries(query)
        queries = [query]
        print("queries:===========", queries)
        retrieved_lists = [self.retriever.invoke(q) for q in queries]
        # print("retrived_lists:=============", retrieved_lists)
        fused_docs = self.reciprocal_rank_fusion(retrieved_lists)
        # print("fused_docs========", fused_docs)
        # Combine both text and URLs
        context = "\n\n".join([
            f"{doc.page_content}\n\n[Source]({doc.metadata.get('source', '')})"
            for doc in fused_docs
        ])
        print(f"📚 Retrieved {len(fused_docs)} fused documents from RRF")
        return context

    
    def process_response(self, content):
        """Format the response for better readability"""
        if "step" in content and "content" in content:
            step = content["step"].lower()
            step_content = content["content"]
            
            if step == "plan":
                return f"🧠 PLANNING: {step_content}"
            elif step == "analyze":
                return f"🔍 ANALYZING: {step_content}"
            elif step == "retrieve":
                return f"📚 RETRIEVING: {step_content}"
            elif step == "synthesize":
                return f"🧩 SYNTHESIZING: {step_content}"
            elif step == "output":
                return f"\n📝 ANSWER:\n{step_content}"
        
        return content

    def setup_retriever(self):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        qdrant_url = "https://360dc222-b8e8-4077-ae28-8c8e6d81d4a9.us-west-1-0.aws.cloud.qdrant.io"
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        # Create Qdrant client
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        # Push documents only if collection doesn't exist
        try:
            qdrant_client.get_collection("braeden_docs")
            print("✅ Qdrant collection already exists. Skipping upload.")
        except Exception:
            print("❗ Qdrant collection not found. Uploading documents...")
            data = self.load_sitemap()
            splitted_docs = self.split_text(data)

            Qdrant.from_documents(
                documents=splitted_docs,
                embedding=embeddings,
                collection_name="braeden_docs",
                url=qdrant_url,
                api_key=qdrant_api_key,
                batch_size=16
            )

        # Load vectorstore from cloud
        vectorstore = Qdrant(
            client=qdrant_client,
            collection_name="braeden_docs",
            embeddings=embeddings
        )

        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})
        return retriever

    def run(self):
        print("\n" + "=" * 60)
        print("🚀 BraedenBot Documentation Assistant 🚀")
        print("=" * 60)
        print("\nA RAG-powered assistant for Braeden documentation")
        print("\nType 'exit' to quit the assistant")
        print("=" * 60 + "\n")
        
        try:
            while True:
                query = input("➤ Ask about Braeden docs: ")
                
                if query.lower() in ["exit", "quit"]:
                    print("\n👋 Goodbye! BraedenBot Documentation Assistant is shutting down.")
                    break
                
                context = self.get_context_for_query(query)
                self.messages.append({
                    "role": "user", 
                    "content": query
                })
                self.messages.append({
                    "role": "assistant", 
                    "content": f"Relevant Braeden documentation context:\n\n{context}"
                })
                
                conversation_active = True
                current_step = None
                
                print("\n⏳ Processing your query...\n")
                
                while conversation_active:
                    try:
                        response = self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            response_format={"type": "json_object"},
                            messages=self.messages,
                        )
                        
                        try:
                            response_content = response.choices[0].message.content


                            parsed_output = json.loads(response_content)
                            
                            self.messages.append({
                                "role": "assistant",
                                "content": response_content
                            })
                            
                            step = parsed_output.get("step", "").lower()
                            
                            if step != current_step:
                                current_step = step
                                formatted_output = self.process_response(parsed_output)
                                print(formatted_output)
                            
                            if step == "output":
                                conversation_active = False
                            
                        except json.JSONDecodeError:
                            print("❌ Error: Invalid JSON response from API")
                            print(f"Raw response: {response_content[:100]}...")
                            conversation_active = False
                            
                    except Exception as e:
                        print(f"❌ Error: {str(e)}")
                        conversation_active = False
                
                print("\n" + "-" * 60 + "\n")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye! BraedenBot Documentation Assistant is shutting down.")

if __name__ == "__main__":
    bot = BraedenBot()
    bot.run()