import app
payload = {
    'duration': 0,
    'protocol_type': 'tcp',
    'service': 'http',
    'flag': 'SF',
    'src_bytes': 232,
    'dst_bytes': 8153,
    'count': 5,
    'srv_count': 5,
    'serror_rate': 0.0,
    'rerror_rate': 0.0,
    'same_srv_rate': 1.0,
    'diff_srv_rate': 0.0,
}
vec = app._build_feature_vector(payload)
print(len(vec))
print(vec[:10])
