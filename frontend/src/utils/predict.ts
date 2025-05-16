import type { NormalizedLandmark } from "@mediapipe/hands";

export async function sendLandmarksToBackend(
    landmarks: NormalizedLandmark[]
): Promise<string | null>{
    if(landmarks.length !== 21){
        console.warn("Should have 21 landmarks, but there's only ", landmarks.length);
        return null;
    }

    const flattened = landmarks.flatMap((lm) => [lm.x, lm.y, lm.z]);
    try{
        const res = await fetch(`${import.meta.env.VITE_API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-type": "application/json",
            },
            body: JSON.stringify({landmarks: flattened}),
        })

        if(!res.ok){
            console.error("Something happened: ", res.statusText);
            return null;
        }
        const data = await res.json();

        if(!data.letter){
            console.warn("No letter in backend response, it's showing ", data);
            return null;
        }
        return data.letter;
    }
    catch(err){
        console.error("Fetch failed!", err);
        return null;
    }
}