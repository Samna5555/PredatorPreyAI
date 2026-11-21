from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

SIZE = 7
EXIT = (6, 6)
WALLS = {
    (0, 3), (1, 3), (2, 3),
    (3, 1), (3, 2), (3, 4), (3, 5),
    (4, 5), (5, 2), (5, 3)
}
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]


def valid_moves(position):
    r, c = position
    result = []
    for dr, dc in ACTIONS:
        nxt = (r + dr, c + dc)
        if 0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE and nxt not in WALLS:
            result.append(nxt)
    return result


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def evaluate(predator, prey):
    if predator == prey:
        return 1000
    if prey == EXIT:
        return -1000
    # Higher is better for predator: closer to prey; discourage prey being near exit.
    return -distance(predator, prey) - 0.4 * distance(prey, EXIT)


def minimax(predator, prey, depth, maximizing, alpha, beta, pruning, stats):
    stats["nodes"] += 1
    if predator == prey:
        return 1000
    if prey == EXIT:
        return -1000
    if depth <= 0:
        return evaluate(predator, prey)

    if maximizing:
        best = float("-inf")
        for move in valid_moves(predator):
            score = minimax(move, prey, depth - 1, False, alpha, beta, pruning, stats)
            best = max(best, score)
            if pruning:
                alpha = max(alpha, best)
                if beta <= alpha:
                    stats["pruned"] += 1
                    break
        return best
    else:
        best = float("inf")
        for move in valid_moves(prey):
            score = minimax(predator, move, depth - 1, True, alpha, beta, pruning, stats)
            best = min(best, score)
            if pruning:
                beta = min(beta, best)
                if beta <= alpha:
                    stats["pruned"] += 1
                    break
        return best


def choose_predator_move(predator, prey, depth, pruning):
    best_move = predator
    best_score = float("-inf")
    stats = {"nodes": 0, "pruned": 0}
    scores = []

    for move in valid_moves(predator):
        score = minimax(
            move, prey, max(0, depth - 1), False,
            float("-inf"), float("inf"), pruning, stats
        )
        scores.append({"position": list(move), "score": score})
        if score > best_score:
            best_score = score
            best_move = move

    return best_move, scores, stats


@app.route("/")
def home():
    return "Predator and Prey AI Backend is Running!"


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/move", methods=["POST"])
def get_ai_move():
    data = request.get_json(force=True)
    predator = tuple(data["predator"])
    prey = tuple(data["prey"])
    depth = max(1, min(int(data.get("depth", 3)), 5))
    pruning = bool(data.get("pruning", True))

    move, scores, stats = choose_predator_move(predator, prey, depth, pruning)
    return jsonify({
        "predator": list(move),
        "scores": scores,
        "nodes": stats["nodes"],
        "pruned": stats["pruned"],
        "algorithm": "Minimax with Alpha-Beta Pruning",
        "depth": depth
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
