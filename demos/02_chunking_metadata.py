from rag_core import CORPUS


print("Chunk metadata matters because retrieved text must be traceable.")
for doc in CORPUS[:5]:
    print(f"- id={doc.id} source={doc.source} tags={','.join(doc.tags)} title={doc.title}")

print("\nPractice task:")
print("Replace CORPUS with your own Markdown files and preserve title path, source path, and updated_at.")

