"""
Task definitions for the experiment.

We use a curated set of coding tasks that test each sub-task type.
Each task has a known ground-truth solution for evaluation.
These are designed to be representative of SWE-bench-style issues
while being self-contained (no Docker/repo checkout needed).
"""

import json
import os

# Each task simulates a real-world coding issue with sub-tasks.
# The agent must: explore the code, understand it, localize the bug,
# generate a patch, write a test, and verify.

CODING_TASKS = [
    # ── Category: Bug Fixes ──────────────────────────────────
    {
        "id": "bugfix_001",
        "category": "bugfix",
        "title": "Fix off-by-one error in pagination",
        "description": "The `paginate` function returns one fewer item than expected on the last page.",
        "code": '''def paginate(items, page, per_page=10):
    """Return items for a given page number (1-indexed)."""
    start = (page - 1) * per_page
    end = start + per_page - 1  # BUG: should be start + per_page
    return items[start:end]

def total_pages(items, per_page=10):
    return (len(items) + per_page - 1) // per_page''',
        "test_code": '''def test_paginate():
    items = list(range(25))
    assert paginate(items, 1) == list(range(10))
    assert paginate(items, 2) == list(range(10, 20))
    assert paginate(items, 3) == list(range(20, 25))
    assert len(paginate(items, 1)) == 10''',
        "expected_fix": "end = start + per_page",
        "difficulty": 0.2,
    },
    {
        "id": "bugfix_002",
        "category": "bugfix",
        "title": "Fix dictionary key error in user lookup",
        "description": "The `get_user_email` function crashes when the user has no email field instead of returning None.",
        "code": '''def get_user_email(users_db, user_id):
    """Return email for a user, or None if not found."""
    user = users_db.get(user_id)
    return user["email"]  # BUG: crashes if user is None or has no email

def get_user_name(users_db, user_id):
    user = users_db.get(user_id)
    if user is None:
        return None
    return user.get("name")''',
        "test_code": '''def test_get_user_email():
    db = {"u1": {"name": "Alice", "email": "a@b.com"}, "u2": {"name": "Bob"}}
    assert get_user_email(db, "u1") == "a@b.com"
    assert get_user_email(db, "u2") is None
    assert get_user_email(db, "u3") is None''',
        "expected_fix": 'return user.get("email") if user else None',
        "difficulty": 0.15,
    },
    {
        "id": "bugfix_003",
        "category": "bugfix",
        "title": "Fix race condition in counter increment",
        "description": "The `SafeCounter` class is not actually thread-safe despite its name.",
        "code": '''import threading

class SafeCounter:
    def __init__(self):
        self.value = 0

    def increment(self):
        # BUG: not thread-safe, needs a lock
        current = self.value
        self.value = current + 1

    def get(self):
        return self.value''',
        "test_code": '''import threading

def test_safe_counter():
    counter = SafeCounter()
    threads = []
    for _ in range(100):
        t = threading.Thread(target=lambda: [counter.increment() for _ in range(100)])
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert counter.get() == 10000''',
        "expected_fix": "Use threading.Lock in __init__ and acquire/release in increment",
        "difficulty": 0.4,
    },
    {
        "id": "bugfix_004",
        "category": "bugfix",
        "title": "Fix incorrect timezone conversion",
        "description": "The `to_utc` function doesn't handle daylight saving time correctly.",
        "code": '''from datetime import datetime, timedelta

def to_utc(dt_str, offset_hours):
    """Convert a datetime string with known UTC offset to UTC.
    E.g., to_utc('2024-03-10 14:00:00', -5) for EST."""
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    # BUG: should subtract offset to get UTC, not add
    return dt + timedelta(hours=offset_hours)''',
        "test_code": '''def test_to_utc():
    # 2pm EST (UTC-5) should be 7pm UTC
    result = to_utc("2024-03-10 14:00:00", -5)
    assert result.hour == 19
    # 10am IST (UTC+5:30) ... using +5 for simplicity
    result2 = to_utc("2024-03-10 10:00:00", 5)
    assert result2.hour == 5''',
        "expected_fix": "return dt - timedelta(hours=offset_hours)",
        "difficulty": 0.3,
    },
    {
        "id": "bugfix_005",
        "category": "bugfix",
        "title": "Fix binary search returning wrong index",
        "description": "The `binary_search` function sometimes returns an index pointing to the wrong element.",
        "code": '''def binary_search(arr, target):
    """Return index of target in sorted array, or -1 if not found."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] < target:
            low = mid  # BUG: should be mid + 1
        elif arr[mid] > target:
            high = mid  # BUG: should be mid - 1
        else:
            return mid
    return -1''',
        "test_code": '''def test_binary_search():
    arr = [1, 3, 5, 7, 9, 11, 13]
    assert binary_search(arr, 7) == 3
    assert binary_search(arr, 1) == 0
    assert binary_search(arr, 13) == 6
    assert binary_search(arr, 4) == -1
    assert binary_search(arr, 0) == -1''',
        "expected_fix": "low = mid + 1 and high = mid - 1",
        "difficulty": 0.25,
    },
    {
        "id": "bugfix_006",
        "category": "bugfix",
        "title": "Fix SQL injection vulnerability in query builder",
        "description": "The `build_query` function uses string formatting instead of parameterized queries.",
        "code": '''def build_query(table, filters):
    """Build a SELECT query with filters.
    filters is a dict like {'name': 'Alice', 'age': 30}."""
    query = f"SELECT * FROM {table}"  # BUG: table name injection
    if filters:
        conditions = []
        for key, value in filters.items():
            conditions.append(f"{key} = '{value}'")  # BUG: value injection
        query += " WHERE " + " AND ".join(conditions)
    return query''',
        "test_code": '''def test_build_query_safe():
    # Should use parameterized queries
    query, params = build_query("users", {"name": "Alice", "age": 30})
    assert "?" in query or "%s" in query  # parameterized
    assert "Alice" not in query  # value not interpolated
    assert params == ["Alice", 30] or params == ("Alice", 30)''',
        "expected_fix": "Return (query_with_placeholders, params_tuple) using ? or %s",
        "difficulty": 0.45,
    },
    {
        "id": "bugfix_007",
        "category": "bugfix",
        "title": "Fix memory leak in cache with no eviction",
        "description": "The `Cache` class grows without bound. Add LRU eviction.",
        "code": '''class Cache:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        # BUG: never evicts old entries when max_size is reached
        self.data[key] = value''',
        "test_code": '''def test_cache_eviction():
    cache = Cache(max_size=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.get("a")  # access a so it's recently used
    cache.put("d", 4)  # should evict least recently used (b)
    assert cache.get("a") == 1
    assert cache.get("b") is None  # evicted
    assert cache.get("d") == 4''',
        "expected_fix": "Use OrderedDict or track access order for LRU eviction in put()",
        "difficulty": 0.5,
    },
    {
        "id": "bugfix_008",
        "category": "bugfix",
        "title": "Fix incorrect CSV parsing with quoted fields",
        "description": "The `parse_csv_line` function fails when fields contain commas inside quotes.",
        "code": '''def parse_csv_line(line):
    """Parse a single CSV line into fields."""
    # BUG: naive split doesn't handle quoted fields with commas
    return line.split(",")''',
        "test_code": '''def test_parse_csv():
    assert parse_csv_line('a,b,c') == ['a', 'b', 'c']
    assert parse_csv_line('"hello, world",b,c') == ['hello, world', 'b', 'c']
    assert parse_csv_line('a,"b,c",d') == ['a', 'b,c', 'd']
    assert parse_csv_line('"a""b",c') == ['a"b', 'c']''',
        "expected_fix": "Implement proper CSV parsing respecting quoted fields",
        "difficulty": 0.55,
    },
    {
        "id": "bugfix_009",
        "category": "bugfix",
        "title": "Fix recursive fibonacci stack overflow",
        "description": "The `fibonacci` function overflows the stack for large inputs. Add memoization.",
        "code": '''def fibonacci(n):
    """Return the nth Fibonacci number."""
    # BUG: exponential time, stack overflow for n > ~1000
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)''',
        "test_code": '''def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(10) == 55
    assert fibonacci(50) == 12586269025
    assert fibonacci(100) == 354224848179261915075  # must handle large n''',
        "expected_fix": "Add memoization via functools.lru_cache or iterative approach",
        "difficulty": 0.25,
    },
    {
        "id": "bugfix_010",
        "category": "bugfix",
        "title": "Fix incorrect merge of sorted lists",
        "description": "The `merge_sorted` function produces incorrect output when one list is exhausted before the other.",
        "code": '''def merge_sorted(list1, list2):
    """Merge two sorted lists into one sorted list."""
    result = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    # BUG: missing the remaining elements from whichever list isn't exhausted
    return result''',
        "test_code": '''def test_merge_sorted():
    assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
    assert merge_sorted([1, 2], [3, 4, 5, 6]) == [1, 2, 3, 4, 5, 6]
    assert merge_sorted([], [1, 2]) == [1, 2]
    assert merge_sorted([1], []) == [1]''',
        "expected_fix": "Add result.extend(list1[i:]) and result.extend(list2[j:])",
        "difficulty": 0.15,
    },
    # ── Category: Feature Implementation ─────────────────────
    {
        "id": "feature_001",
        "category": "feature",
        "title": "Implement a retry decorator with exponential backoff",
        "description": "Create a decorator that retries a function up to N times with exponential backoff on exception.",
        "code": '''# No existing implementation. Write from scratch.
# Requirements:
# - retry(max_retries=3, base_delay=1.0) decorator
# - Exponential backoff: delay doubles each retry
# - Re-raises the last exception if all retries fail
# - Should work with any function signature
''',
        "test_code": '''import time

def test_retry_decorator():
    call_count = 0
    @retry(max_retries=3, base_delay=0.01)
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("not yet")
        return "success"

    result = flaky_function()
    assert result == "success"
    assert call_count == 3

def test_retry_exhausted():
    @retry(max_retries=2, base_delay=0.01)
    def always_fails():
        raise RuntimeError("fail")

    try:
        always_fails()
        assert False, "Should have raised"
    except RuntimeError:
        pass''',
        "expected_fix": "Implement retry decorator with time.sleep(base_delay * 2**attempt)",
        "difficulty": 0.4,
    },
    {
        "id": "feature_002",
        "category": "feature",
        "title": "Implement a rate limiter class",
        "description": "Create a token-bucket rate limiter that allows N requests per second.",
        "code": '''# No existing implementation. Write from scratch.
# Requirements:
# - RateLimiter(rate=10, per=1.0) allows 10 calls per 1 second
# - limiter.acquire() blocks until a token is available
# - limiter.try_acquire() returns True/False without blocking
# - Thread-safe
''',
        "test_code": '''import time

def test_rate_limiter():
    limiter = RateLimiter(rate=5, per=1.0)
    # Should allow 5 rapid calls
    for _ in range(5):
        assert limiter.try_acquire() is True
    # 6th should fail (bucket exhausted)
    assert limiter.try_acquire() is False

def test_rate_limiter_refill():
    limiter = RateLimiter(rate=2, per=0.1)
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False
    time.sleep(0.15)  # wait for refill
    assert limiter.try_acquire() is True''',
        "expected_fix": "Token bucket with timestamp tracking and refill logic",
        "difficulty": 0.55,
    },
    {
        "id": "feature_003",
        "category": "feature",
        "title": "Implement a trie (prefix tree) for autocomplete",
        "description": "Create a Trie class supporting insert, search, and prefix-based suggestions.",
        "code": '''# No existing implementation. Write from scratch.
# Requirements:
# - trie.insert(word) adds a word
# - trie.search(word) returns True if word exists
# - trie.starts_with(prefix) returns True if any word starts with prefix
# - trie.autocomplete(prefix, limit=5) returns up to `limit` words with that prefix
''',
        "test_code": '''def test_trie():
    trie = Trie()
    for word in ["apple", "app", "application", "bat", "ball", "banana"]:
        trie.insert(word)

    assert trie.search("app") is True
    assert trie.search("ap") is False
    assert trie.starts_with("app") is True
    assert trie.starts_with("xyz") is False

    suggestions = trie.autocomplete("app")
    assert set(suggestions) == {"app", "apple", "application"}

    suggestions2 = trie.autocomplete("ba", limit=2)
    assert len(suggestions2) == 2''',
        "expected_fix": "Implement TrieNode with children dict and is_end flag, DFS for autocomplete",
        "difficulty": 0.5,
    },
    {
        "id": "feature_004",
        "category": "feature",
        "title": "Implement an event emitter / pub-sub system",
        "description": "Create an EventEmitter class similar to Node.js EventEmitter.",
        "code": '''# No existing implementation. Write from scratch.
# Requirements:
# - emitter.on(event, callback) registers a listener
# - emitter.off(event, callback) removes a listener
# - emitter.emit(event, *args) calls all listeners with args
# - emitter.once(event, callback) listener that fires only once
''',
        "test_code": '''def test_event_emitter():
    emitter = EventEmitter()
    results = []

    def on_data(x):
        results.append(x)

    emitter.on("data", on_data)
    emitter.emit("data", 1)
    emitter.emit("data", 2)
    assert results == [1, 2]

    emitter.off("data", on_data)
    emitter.emit("data", 3)
    assert results == [1, 2]  # no change

def test_once():
    emitter = EventEmitter()
    results = []
    emitter.once("ping", lambda: results.append("pong"))
    emitter.emit("ping")
    emitter.emit("ping")
    assert results == ["pong"]  # only once''',
        "expected_fix": "Dict of event->list of callbacks, once() wraps callback to auto-remove",
        "difficulty": 0.35,
    },
    {
        "id": "feature_005",
        "category": "feature",
        "title": "Implement JSON schema validator",
        "description": "Create a function that validates a JSON object against a simple schema.",
        "code": '''# No existing implementation. Write from scratch.
# Requirements:
# - validate(data, schema) returns (True, []) or (False, [list of errors])
# - Schema supports: {"type": "string|int|float|bool|list|dict"}
# - Schema supports: {"required": ["field1", "field2"]}
# - Schema supports: {"properties": {"field": {sub_schema}}}
''',
        "test_code": '''def test_validate():
    schema = {
        "type": "dict",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "int"},
            "email": {"type": "string"},
        }
    }
    ok, errors = validate({"name": "Alice", "age": 30}, schema)
    assert ok is True
    assert errors == []

    ok2, errors2 = validate({"name": "Bob"}, schema)
    assert ok2 is False
    assert any("age" in e for e in errors2)

    ok3, errors3 = validate({"name": 123, "age": "thirty"}, schema)
    assert ok3 is False
    assert len(errors3) >= 2''',
        "expected_fix": "Recursive validation checking type, required fields, and nested properties",
        "difficulty": 0.5,
    },
    # ── Category: Refactoring ────────────────────────────────
    {
        "id": "refactor_001",
        "category": "refactor",
        "title": "Refactor nested conditionals into early returns",
        "description": "Simplify the deeply nested `process_order` function using guard clauses.",
        "code": '''def process_order(order):
    """Process an order and return status message."""
    result = None
    if order is not None:
        if order.get("status") == "pending":
            if order.get("items") and len(order["items"]) > 0:
                if order.get("payment"):
                    if order["payment"].get("verified"):
                        total = sum(item["price"] * item["qty"] for item in order["items"])
                        if total > 0:
                            result = {"status": "processed", "total": total}
                        else:
                            result = {"status": "error", "message": "invalid total"}
                    else:
                        result = {"status": "error", "message": "payment not verified"}
                else:
                    result = {"status": "error", "message": "no payment info"}
            else:
                result = {"status": "error", "message": "no items"}
        else:
            result = {"status": "error", "message": "order not pending"}
    else:
        result = {"status": "error", "message": "no order"}
    return result''',
        "test_code": '''def test_process_order():
    good_order = {
        "status": "pending",
        "items": [{"price": 10, "qty": 2}, {"price": 5, "qty": 1}],
        "payment": {"verified": True}
    }
    result = process_order(good_order)
    assert result["status"] == "processed"
    assert result["total"] == 25

    assert process_order(None)["message"] == "no order"
    assert process_order({"status": "shipped"})["message"] == "order not pending"
    assert process_order({"status": "pending", "items": []})["message"] == "no items"''',
        "expected_fix": "Use early returns: if order is None: return error; if status != pending: return error; etc.",
        "difficulty": 0.3,
    },
    {
        "id": "refactor_002",
        "category": "refactor",
        "title": "Extract duplicated validation logic into a validator class",
        "description": "Three functions have copy-pasted email/phone validation. Extract into a reusable Validator.",
        "code": '''import re

def register_user(name, email, phone):
    # Duplicated validation
    if not re.match(r"^[\\w.-]+@[\\w.-]+\\.\\w+$", email):
        raise ValueError("Invalid email")
    if not re.match(r"^\\+?\\d{10,15}$", phone):
        raise ValueError("Invalid phone")
    return {"name": name, "email": email, "phone": phone, "type": "user"}

def register_admin(name, email, phone, access_level):
    # Same validation copy-pasted
    if not re.match(r"^[\\w.-]+@[\\w.-]+\\.\\w+$", email):
        raise ValueError("Invalid email")
    if not re.match(r"^\\+?\\d{10,15}$", phone):
        raise ValueError("Invalid phone")
    return {"name": name, "email": email, "phone": phone, "type": "admin", "access": access_level}

def update_contact(email, phone):
    # Same validation again
    if not re.match(r"^[\\w.-]+@[\\w.-]+\\.\\w+$", email):
        raise ValueError("Invalid email")
    if not re.match(r"^\\+?\\d{10,15}$", phone):
        raise ValueError("Invalid phone")
    return {"email": email, "phone": phone}''',
        "test_code": '''def test_validator():
    v = Validator()
    assert v.validate_email("test@example.com") is True
    assert v.validate_email("not-an-email") is False
    assert v.validate_phone("+1234567890") is True
    assert v.validate_phone("abc") is False

def test_register_uses_validator():
    result = register_user("Alice", "a@b.com", "+1234567890")
    assert result["type"] == "user"
    try:
        register_user("Alice", "bad-email", "+1234567890")
        assert False
    except ValueError:
        pass''',
        "expected_fix": "Create Validator class with validate_email/validate_phone, use in all three functions",
        "difficulty": 0.35,
    },
    # ── Category: Multi-step / Complex ───────────────────────
    {
        "id": "complex_001",
        "category": "complex",
        "title": "Implement a complete URL shortener service",
        "description": "Build a URLShortener class with encode, decode, and click tracking.",
        "code": '''# No existing implementation.
# Requirements:
# - shortener.encode(url) returns a short code (6 chars, base62)
# - shortener.decode(code) returns the original URL
# - shortener.click(code) increments click count and returns URL
# - shortener.stats(code) returns {"url": ..., "clicks": N, "created": datetime}
# - Handle duplicate URLs (return same code)
''',
        "test_code": '''def test_url_shortener():
    s = URLShortener()
    code1 = s.encode("https://example.com")
    assert len(code1) == 6
    assert s.decode(code1) == "https://example.com"

    # Same URL should return same code
    code2 = s.encode("https://example.com")
    assert code1 == code2

    # Click tracking
    s.click(code1)
    s.click(code1)
    stats = s.stats(code1)
    assert stats["clicks"] == 2
    assert stats["url"] == "https://example.com"

    # Unknown code
    assert s.decode("xxxxxx") is None''',
        "expected_fix": "Hash-based encoding with base62, dual dicts for bidirectional lookup, counter tracking",
        "difficulty": 0.6,
    },
    {
        "id": "complex_002",
        "category": "complex",
        "title": "Implement a task scheduler with dependencies",
        "description": "Build a TaskScheduler that resolves dependencies and returns execution order.",
        "code": '''# No existing implementation.
# Requirements:
# - scheduler.add_task(name, dependencies=[]) adds a task
# - scheduler.get_order() returns topological order
# - Detect circular dependencies and raise an error
# - scheduler.get_ready() returns tasks with all deps satisfied
''',
        "test_code": '''def test_scheduler():
    s = TaskScheduler()
    s.add_task("build", dependencies=["compile"])
    s.add_task("compile", dependencies=["parse"])
    s.add_task("parse", dependencies=[])
    s.add_task("test", dependencies=["build"])
    s.add_task("deploy", dependencies=["test", "build"])

    order = s.get_order()
    assert order.index("parse") < order.index("compile")
    assert order.index("compile") < order.index("build")
    assert order.index("build") < order.index("test")
    assert order.index("build") < order.index("deploy")

def test_circular():
    s = TaskScheduler()
    s.add_task("a", dependencies=["b"])
    s.add_task("b", dependencies=["a"])
    try:
        s.get_order()
        assert False, "Should detect cycle"
    except ValueError:
        pass''',
        "expected_fix": "Topological sort using Kahn's algorithm or DFS, cycle detection",
        "difficulty": 0.55,
    },
    {
        "id": "complex_003",
        "category": "complex",
        "title": "Implement a simple expression evaluator",
        "description": "Build a calculator that evaluates mathematical expressions with +, -, *, /, and parentheses.",
        "code": '''# No existing implementation.
# Requirements:
# - evaluate("2 + 3") == 5
# - evaluate("(2 + 3) * 4") == 20
# - Respect operator precedence: * and / before + and -
# - Handle nested parentheses
# - Raise ValueError for invalid expressions
''',
        "test_code": '''def test_evaluate():
    assert evaluate("2 + 3") == 5
    assert evaluate("10 - 4") == 6
    assert evaluate("2 * 3 + 1") == 7
    assert evaluate("2 + 3 * 4") == 14  # precedence
    assert evaluate("(2 + 3) * 4") == 20
    assert evaluate("((1 + 2) * (3 + 4))") == 21
    assert evaluate("10 / 2") == 5.0

def test_evaluate_errors():
    try:
        evaluate("2 + + 3")
        assert False
    except ValueError:
        pass''',
        "expected_fix": "Recursive descent parser or shunting-yard algorithm",
        "difficulty": 0.7,
    },
    {
        "id": "complex_004",
        "category": "complex",
        "title": "Implement a diff algorithm for text comparison",
        "description": "Build a function that computes line-by-line diff between two texts, similar to Unix diff.",
        "code": '''# No existing implementation.
# Requirements:
# - diff(text1, text2) returns list of changes
# - Each change: {"type": "add"|"remove"|"keep", "line": str, "line_num": int}
# - Uses longest common subsequence (LCS) algorithm
# - Output should be minimal (fewest changes)
''',
        "test_code": '''def test_diff():
    text1 = "line1\\nline2\\nline3\\nline4"
    text2 = "line1\\nline2_modified\\nline3\\nline5"
    changes = diff(text1, text2)

    types = [c["type"] for c in changes]
    assert "keep" in types  # line1 and line3 kept
    assert "remove" in types  # line2 and line4 removed
    assert "add" in types  # line2_modified and line5 added

    # Identical texts should have no changes
    same = diff("a\\nb", "a\\nb")
    assert all(c["type"] == "keep" for c in same)''',
        "expected_fix": "LCS-based diff algorithm comparing line arrays",
        "difficulty": 0.65,
    },
    {
        "id": "complex_005",
        "category": "complex",
        "title": "Implement a simple HTTP router with path parameters",
        "description": "Build a Router class that matches URL paths to handler functions, supporting path parameters.",
        "code": '''# No existing implementation.
# Requirements:
# - router.add_route("GET", "/users/:id", handler)
# - router.match("GET", "/users/42") returns (handler, {"id": "42"})
# - Support multiple HTTP methods
# - Return None for no match
# - Handle static and parameterized segments
''',
        "test_code": '''def test_router():
    router = Router()

    def get_user(params): return f"user {params['id']}"
    def list_users(params): return "all users"
    def create_user(params): return "created"

    router.add_route("GET", "/users", list_users)
    router.add_route("GET", "/users/:id", get_user)
    router.add_route("POST", "/users", create_user)

    handler, params = router.match("GET", "/users/42")
    assert handler(params) == "user 42"
    assert params == {"id": "42"}

    handler2, params2 = router.match("GET", "/users")
    assert handler2(params2) == "all users"

    assert router.match("DELETE", "/users") is None''',
        "expected_fix": "Split path into segments, match static/param segments, extract param values",
        "difficulty": 0.5,
    },
]


def get_tasks(n=None):
    """Return the task list, optionally limited to n tasks."""
    tasks = CODING_TASKS
    if n is not None:
        tasks = tasks[:n]
    return tasks
