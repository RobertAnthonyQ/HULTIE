from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("C:\\Users\diego\PycharmProjects\HultieChatbot\Hultie\data\\UP_BASES.pdf")


print(result.text_content)