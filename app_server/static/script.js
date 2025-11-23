async function sendLotto() {
    const moneyInput = document.getElementById("money");
    const winningInput = document.getElementById("winning");
    const bonusInput = document.getElementById("bonus");
    const resultEl = document.getElementById("result");
    const statusText = document.getElementById("statusText");
    const submitBtn = document.getElementById("submitBtn");

    const money = moneyInput.value;
    const winning = winningInput.value;
    const bonus = bonusInput.value;

    if (!money || !winning || !bonus) {
        statusText.textContent = "모든 값을 입력해주세요.";
        return;
    }

    try {
        submitBtn.disabled = true;
        statusText.textContent = "모델 실행 중...";
        resultEl.textContent = "";

        const res = await fetch("/lotto", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                money: Number(money),
                winning: winning,
                bonus: Number(bonus)
            })
        });

        if (!res.ok) {
            statusText.textContent = "요청 실패";
            resultEl.textContent = `HTTP Error ${res.status}`;
            submitBtn.disabled = false;
            return;
        }

        const data = await res.json();
        statusText.textContent = "완료";
        resultEl.textContent = data.result ?? "";
    } catch (e) {
        statusText.textContent = "에러 발생";
        resultEl.textContent = e?.message ?? String(e);
    } finally {
        submitBtn.disabled = false;
    }
}