from src.agents.orchestrator import OrchestratorAgent


def test_summary(orchestrator,url):

    state = orchestrator.run(video_url=url)

    print("\nSUMMARY:")
    print(state.get("summary"))

    print("\nTRACE:")
    print(state.get("agent_trace"))

    assert state is not None


def test_qa(orchestrator,url):

    state = orchestrator.run(
        video_url=url,
        question="Explain main idea"
    )

    print("\nANSWER:")
    print(state.get("answer"))

    print("\nTRACE:")
    print(state.get("agent_trace"))

    assert state is not None


def test_invalid_url(orchestrator):

    state = orchestrator.run(
        video_url="invalid_url"
    )

    print("\nERROR:")
    print(state.get("error"))

    assert state.get("error") is not None


if __name__ == "__main__":

    orchestrator = OrchestratorAgent()

    url = "https://youtu.be/56BXHCkngss?si=oYfW8l6iN8EUOMKZ"

    test_summary(orchestrator,url)

    test_qa(orchestrator,url)

    # test_invalid_url(orchestrator)