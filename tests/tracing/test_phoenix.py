from utu.tracing.phoenix_utils import PhoenixUtils

phoenix_utils = PhoenixUtils()


def test_get_trace_url():
    print(phoenix_utils.get_trace_url_by_id("trace_a4fca2ae5c95463bbd144412d3051db0"))
