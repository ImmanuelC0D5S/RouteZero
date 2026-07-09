from routezero.conversation import ConversationStore, PipelineState

def test_new_session():
    store = ConversationStore()
    session_id = store.new_session()
    assert len(session_id) == 32  # uuid4 hex
    assert store.get_history(session_id) == []

def test_append_and_get_history():
    store = ConversationStore(window_size=5)
    session_id = store.new_session()
    
    store.append_turn(session_id, "user", "Hello")
    store.append_turn(session_id, "assistant", "Hi there!")
    
    history = store.get_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"

def test_window_size_trimming():
    store = ConversationStore(window_size=2)
    session_id = store.new_session()
    
    store.append_turn(session_id, "user", "msg1")
    store.append_turn(session_id, "user", "msg2")
    store.append_turn(session_id, "user", "msg3")
    
    history = store.get_history(session_id)
    assert len(history) == 2
    assert history[0]["content"] == "msg2"

def test_build_contextual_prompt():
    store = ConversationStore()
    session_id = store.new_session()
    store.append_turn(session_id, "user", "What is AI?")
    store.append_turn(session_id, "assistant", "AI is machine intelligence.")
    
    prompt = store.build_contextual_prompt(session_id, "Tell me more")
    assert "[System]" in prompt
    assert "[Conversation history]" in prompt
    assert "What is AI?" in prompt
    assert "[Current query]" in prompt
    assert "Tell me more" in prompt

def test_pipeline_state_typeddict():
    state: PipelineState = {
        "user_prompt": "test",
        "prompt_embedding": [],
        "cache_hit": False,
        "routing_target": "local",
        "task_type": "factual",
        "model_response": "",
        "verification_passed": False,
        "execution_latency_ms": 0.0,
        "conversation_id": "abc123",
        "turn_index": 0,
    }
    assert state["user_prompt"] == "test"
