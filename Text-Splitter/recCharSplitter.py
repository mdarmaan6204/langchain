from langchain.text_splitter import RecursiveCharacterTextSplitter , Language


text = """
The Indian Premier League (IPL) has revolutionized cricket since its inception in 2008, transforming the sport into a global entertainment spectacle. This Twenty20 cricket league features eight franchise teams representing major Indian cities, each owned by prominent business personalities and Bollywood celebrities. The IPL follows a unique auction system where teams bid for players, creating a perfect blend of international and domestic talent. With its high-octane matches, strategic timeouts, cheerleaders, and celebrity ownership, the IPL has successfully bridged the gap between sports and entertainment, attracting millions of viewers worldwide and generating billions in revenue.

The tournament's impact extends far beyond cricket, as it has become a significant economic driver for India's sports industry. The IPL has provided a platform for young Indian cricketers to showcase their skills alongside international superstars like Virat Kohli, MS Dhoni, AB de Villiers, and Chris Gayle. The league's innovative marketing strategies, including franchise-based team loyalty, merchandise sales, and strategic partnerships with global brands, have set new standards for sports marketing. Additionally, the IPL has contributed to the development of cricket infrastructure across India, with state-of-the-art stadiums and training facilities being built to accommodate the league's requirements, ultimately elevating the standard of cricket in the country.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300 , 
    chunk_overlap = 0
)

# For Code
splitter2 = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 100,
    chunk_overlap = 0
)

res = splitter2.split_text(text)

print(len(res))
print(res)