# Lotus_base_pretrain.sh object 의 얇은 래퍼. 실제 정의는 그쪽 한 곳에만 있다.
#   bash Lotus_object_pretrain.sh [seed] [gpu]
exec bash "$(dirname "${BASH_SOURCE[0]}")/Lotus_base_pretrain.sh" object "${2:-0}" "${1:-7}";
