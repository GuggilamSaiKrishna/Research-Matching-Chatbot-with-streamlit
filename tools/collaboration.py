from tools.retriever import retrieve_faculty

def collaboration_match(query):

    matches = retrieve_faculty(query, k=2)

    if len(matches) < 2:
        return "No collaboration found."

    return f"""
Suggested Collaboration

Faculty 1:
{matches[0]['name']}

Faculty 2:
{matches[1]['name']}

Reason:
Their research interests are closely related and can collaborate on this topic.
"""