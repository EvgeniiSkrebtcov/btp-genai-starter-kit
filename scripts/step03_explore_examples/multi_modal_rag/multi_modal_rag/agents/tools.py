from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.tools import BaseTool
from langchain_core.language_models import LanguageModelLike
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever


class RAGTool(BaseTool):
    name = "rag_tool"
    description = """
        Useful information about Cities from Wikipedia.
        Input: A question about a city.
        Output: The answer to the question."""
    llm: LanguageModelLike
    retriever: VectorStoreRetriever

    def _run(self, query: str) -> str:
        """Use the tool."""
        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentence maximum and keep the answer concise. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        chain = create_retrieval_chain(self.retriever, question_answer_chain)

        result = chain.invoke({"input": query})

        return result["answer"]
