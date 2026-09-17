let currentDocumentId = null;


const fileInput = document.getElementById(
    "document-file"
);

const uploadButton = document.getElementById(
    "upload-button"
);

const uploadStatus = document.getElementById(
    "upload-status"
);

const documentPanel = document.getElementById(
    "document-panel"
);

const documentName = document.getElementById(
    "document-name"
);

const documentStatus = document.getElementById(
    "document-status"
);

const documentIdElement = document.getElementById(
    "document-id"
);

const processButton = document.getElementById(
    "process-button"
);

const processStatus = document.getElementById(
    "process-status"
);

const questionPanel = document.getElementById(
    "question-panel"
);

const questionInput = document.getElementById(
    "question-input"
);

const askButton = document.getElementById(
    "ask-button"
);

const questionStatus = document.getElementById(
    "question-status"
);

const answerPanel = document.getElementById(
    "answer-panel"
);

const answerHeading = document.getElementById(
    "answer-heading"
);

const answerElement = document.getElementById(
    "answer"
);

const confidenceElement = document.getElementById(
    "confidence"
);

const groundingElement = document.getElementById(
    "grounding"
);

const citationsElement = document.getElementById(
    "citations"
);


function setStatus(
    element,
    message,
    isError = false
) {
    element.textContent = message;
    element.classList.toggle(
        "error",
        isError
    );
}


async function readError(response) {
    try {
        const payload = await response.json();

        if (
            payload &&
            typeof payload.detail === "string"
        ) {
            return payload.detail;
        }
    } catch {
        // Fall through to a generic HTTP message.
    }

    return `Request failed with HTTP ${response.status}.`;
}


function renderAnswer(answer, citations) {
    answerElement.replaceChildren();

    const validCitationIds = new Set(
        citations.map(
            (citation) => citation.citation_id
        )
    );

    const citationPattern =
        /(\[C\d+\])/gi;

    const fragments = answer.split(
        citationPattern
    );

    for (const fragment of fragments) {
        const match = fragment.match(
            /^\[(C\d+)\]$/i
        );

        if (
            match &&
            validCitationIds.has(
                match[1].toUpperCase()
            )
        ) {
            const button =
                document.createElement("button");

            button.type = "button";
            button.className = "citation-link";
            button.textContent = fragment;

            button.addEventListener(
                "click",
                () => {
                    focusCitation(
                        match[1].toUpperCase()
                    );
                }
            );

            answerElement.appendChild(
                button
            );

            continue;
        }

        answerElement.appendChild(
            document.createTextNode(fragment)
        );
    }
}


function focusCitation(citationId) {
    const card = document.getElementById(
        `citation-${citationId}`
    );

    if (!card) {
        return;
    }

    document.querySelectorAll(
        ".citation-card.focused"
    ).forEach(
        (element) => {
            element.classList.remove(
                "focused"
            );
        }
    );

    card.scrollIntoView({
        behavior: "smooth",
        block: "center",
    });

    card.classList.add(
        "focused"
    );

    window.setTimeout(
        () => {
            card.classList.remove(
                "focused"
            );
        },
        1800
    );
}


function renderGrounding(result) {
    groundingElement.replaceChildren();

    if (!result.grounding) {
        return;
    }

    const grounding = result.grounding;

    const status = document.createElement(
        "span"
    );

    status.className =
        grounding.valid
            ? "grounding-valid"
            : "grounding-invalid";

    status.textContent =
        grounding.valid
            ? "Grounded"
            : "Grounding review required";

    const details = document.createElement(
        "span"
    );

    details.textContent =
        ` ${grounding.supported_claim_count}/` +
        `${grounding.claim_count} claims supported`;

    groundingElement.appendChild(
        status
    );

    groundingElement.appendChild(
        details
    );
}


function renderCitations(citations) {
    citationsElement.replaceChildren();

    for (const citation of citations) {
        const card = document.createElement(
            "article"
        );

        card.id =
            `citation-${citation.citation_id}`;

        card.className =
            "citation-card";

        const header =
            document.createElement("div");

        header.className =
            "citation-header";

        const id =
            document.createElement("strong");

        id.textContent =
            citation.citation_id;

        const source =
            document.createElement("span");

        source.textContent =
            citation.source ||
            "Unknown source";

        header.appendChild(id);
        header.appendChild(source);

        const location =
            document.createElement("div");

        location.className =
            "citation-location";

        if (citation.page !== null) {
            location.textContent =
                `Page ${citation.page}`;
        } else {
            location.textContent =
                "Page unavailable";
        }

        const excerpt =
            document.createElement("p");

        excerpt.className =
            "citation-excerpt";

        excerpt.textContent =
            citation.text;

        card.appendChild(header);
        card.appendChild(location);
        card.appendChild(excerpt);

        const section =
            citation.metadata?.section;

        if (section) {
            const sectionElement =
                document.createElement(
                    "div"
                );

            sectionElement.className =
                "citation-section";

            sectionElement.textContent =
                `Section: ${section}`;

            card.appendChild(
                sectionElement
            );
        }

        citationsElement.appendChild(
            card
        );
    }
}


async function uploadDocument() {
    const file = fileInput.files[0];

    if (!file) {
        setStatus(
            uploadStatus,
            "Choose a document first.",
            true
        );
        return;
    }

    const formData = new FormData();

    formData.append(
        "file",
        file
    );

    uploadButton.disabled = true;

    setStatus(
        uploadStatus,
        "Uploading..."
    );

    try {
        const response = await fetch(
            "/documents/upload",
            {
                method: "POST",
                body: formData,
            }
        );

        if (!response.ok) {
            throw new Error(
                await readError(response)
            );
        }

        const record =
            await response.json();

        currentDocumentId =
            record.document_id;

        documentName.textContent =
            record.filename;

        documentStatus.textContent =
            record.status;

        documentIdElement.textContent =
            record.document_id;

        documentPanel.classList.remove(
            "hidden"
        );

        questionPanel.classList.add(
            "hidden"
        );

        answerPanel.classList.add(
            "hidden"
        );

        processButton.disabled =
            false;

        setStatus(
            uploadStatus,
            "Upload complete."
        );

        setStatus(
            processStatus,
            ""
        );
    } catch (error) {
        setStatus(
            uploadStatus,
            error.message,
            true
        );
    } finally {
        uploadButton.disabled = false;
    }
}


async function processDocument() {
    if (!currentDocumentId) {
        return;
    }

    processButton.disabled =
        true;

    setStatus(
        processStatus,
        "Processing and indexing..."
    );

    try {
        const response =
            await fetch(
                `/documents/${currentDocumentId}/process`,
                {
                    method: "POST",
                }
            );

        if (!response.ok) {
            throw new Error(
                await readError(response)
            );
        }

        const result =
            await response.json();

        documentStatus.textContent =
            result.status;

        questionPanel.classList.remove(
            "hidden"
        );

        setStatus(
            processStatus,
            `Ready. Indexed ${result.chunk_count} chunks.`
        );

        questionInput.focus();
    } catch (error) {
        setStatus(
            processStatus,
            error.message,
            true
        );
    } finally {
        processButton.disabled =
            false;
    }
}


async function askQuestion() {
    if (!currentDocumentId) {
        return;
    }

    const question =
        questionInput.value.trim();

    if (!question) {
        setStatus(
            questionStatus,
            "Enter a question first.",
            true
        );
        return;
    }

    askButton.disabled =
        true;

    setStatus(
        questionStatus,
        "Searching evidence and generating answer..."
    );

    try {
        const response =
            await fetch(
                `/documents/${currentDocumentId}/ask`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json",
                    },
                    body: JSON.stringify({
                        question,
                    }),
                }
            );

        if (!response.ok) {
            throw new Error(
                await readError(response)
            );
        }

        const result =
            await response.json();

        answerHeading.textContent =
            result.question;

        confidenceElement.textContent =
            `Confidence ${result.confidence}`;

        renderAnswer(
            result.answer,
            result.citations
        );

        renderGrounding(
            result
        );

        renderCitations(
            result.citations
        );

        answerPanel.classList.remove(
            "hidden"
        );

        setStatus(
            questionStatus,
            `Retrieved ${result.retrieved_count} results.`
        );
    } catch (error) {
        setStatus(
            questionStatus,
            error.message,
            true
        );
    } finally {
        askButton.disabled =
            false;
    }
}


uploadButton.addEventListener(
    "click",
    uploadDocument
);

processButton.addEventListener(
    "click",
    processDocument
);

askButton.addEventListener(
    "click",
    askQuestion
);

questionInput.addEventListener(
    "keydown",
    (event) => {
        if (
            event.key === "Enter" &&
            (event.ctrlKey || event.metaKey)
        ) {
            event.preventDefault();
            askQuestion();
        }
    }
);
