def ask_retrieve(state: AskState):
    """Truy vấn thông tin từ vector DB của user."""
    print(f"🔍 Đang tìm kiếm thông tin cho câu hỏi: {state['question']}")

    results = ask_collection.query(
        query_texts=[state["question"]],
        n_results=5,
        where={"language": state["language"]},
    )

    print('Documents:', results)

    distances = results["distances"][0]

    if not distances or distances[0] > 4.5:
        results["documents"][0] = []

    if state['language'] == 'vi':
        state['language'] = 'Việt Nam'
    else:
        state['language'] = 'English'


    return {"context": results, "language": state["language"]}