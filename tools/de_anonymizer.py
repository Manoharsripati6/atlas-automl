def deanonymize_text(text, metadata):

    mapping = metadata["column_map"]

    for anon, original in mapping.items():

        text = text.replace(
            anon,
            original
        )

    return text