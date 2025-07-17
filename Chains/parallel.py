from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model = 'gemini-2.0-flash',
)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = 'Make a notes of the following text \n {text}',
    input_varaibles = ['text']
)

prompt2 = PromptTemplate(
    template = 'Make a quiz of the following text \n {text}',
    input_varaibles = ['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into single document notes -> {notes} and quiz -> {quiz}',
    input_varaibles = ['quiz' , 'notes']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'notes' : prompt1 | model1 | parser,
        'quiz' : prompt2 | model2 | parser
    }
)

merged_chain = prompt3 | model1 | parser

final_chain = parallel_chain | merged_chain

text = """
A priority queue is a type of queue that arranges elements based on their priority values.

Each element has a priority associated. When we add an item, it is inserted in a position based on its priority.
Elements with higher priority are typically retrieved or removed before elements with lower priority.
Binary heap is the most common method to implement a priority queue. In binary heaps, we have easy access to the min (in min heap) or max (in max heap) and binary heap being a complete binary tree are easily implemented using arrays. Since we use arrays, we have cache friendliness advantage also.
Priority Queue is used in algorithms such as Dijkstra's algorithm, Prim's algorithm, and Huffman Coding.
For example, in the below priority queue, an element with a maximum ASCII value will have the highest priority. The elements with higher priority are served first. 

Types of Priority Queue
Types-of-priority-Queue
Types of Priority Queue
Ascending Order Priority Queue : In this queue, elements with lower values have higher priority. For example, with elements 4, 6, 8, 9, and 10, 4 will be dequeued first since it has the smallest value, and the dequeue operation will return 4.
Descending order Priority Queue : Elements with higher values have higher priority. The root of the heap is the highest element, and it is dequeued first. The queue adjusts by maintaining the heap property after each insertion or deletion.
The following is an example of Descending order Priority Queue.

Priority-Queue-2.webpPriority-Queue-2.webp
How Priority is Determined in a Priority Queue?
In a priority queue, generally, the value of an element is considered for assigning the priority. For example, the element with the highest value is assigned the highest priority and the element with the lowest value is assigned the lowest priority. The reverse case can also be used i.e., the element with the lowest value can be assigned the highest priority. Also, the priority can be assigned according to our needs. 

Operations on a Priority Queue
A typical priority queue supports the following operations:

1) Insertion : If the newly inserted item is of the highest priority, then it is inserted at the top. Otherwise, it is inserted in such a way that it is accessible after all higher priority items are accessed.

2) Deletion : We typically remove the highest priority item which is typically available at the top. Once we remove this item, we need not move next priority item at the top.

3) Peek : This operation only returns the highest priority item (which is typically available at the top) and does not make any change to the priority queue.

Difference between Priority Queue and Normal Queue
There is no priority attached to elements in a queue, the rule of first-in-first-out(FIFO) is implemented whereas, in a priority queue, the elements have a priority. The elements with higher priority are served first.
"""

res = final_chain.invoke({'text': text}) 

print(res)

final_chain.get_graph().print_ascii()

