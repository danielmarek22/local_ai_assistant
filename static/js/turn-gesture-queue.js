export class TurnGestureQueue {
    constructor(maxSize = 3) {
        this.maxSize = Math.max(1, Number(maxSize) || 1);
        this.currentTurnId = null;
        this.items = [];
    }

    setTurn(turnId) {
        const normalized = String(turnId || '').trim() || null;
        if (!normalized || normalized === this.currentTurnId) return false;
        this.currentTurnId = normalized;
        this.items = [];
        return true;
    }

    push(gesture, turnId = null) {
        this.setTurn(turnId);
        this.items.push(gesture);
        if (this.items.length > this.maxSize) {
            this.items.splice(0, this.items.length - this.maxSize);
        }
    }

    peek() {
        return this.items[0] || null;
    }

    shift() {
        return this.items.shift() || null;
    }

    clear() {
        this.items = [];
    }
}
