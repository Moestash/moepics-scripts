import Moepictures from "moepics-api"
import fs from "fs"

const massDeletePosts = async () => {
    const moepics = new Moepictures(process.env.MOEPICTURES_API_KEY!)

    let deleteFolder = process.env.DELETE_FOLDER!
    
    const deleteFiles = fs.readdirSync(deleteFolder).filter((f) => !f.includes(".DS_Store"))
    .sort(new Intl.Collator(undefined, {numeric: true, sensitivity: "base"}).compare)

    for (const file of deleteFiles) {
        let id = file.split("_")[0]?.match(/\d+/)?.[0]!
        if (!id) continue
        if (Number(id) < 0) continue
        // const post = await moepics.posts.get(id)
        // if (post?.type === "comic") console.log(id)
        const result = await moepics.posts.delete(id)
        if (result?.toLowerCase()?.trim() !== "success") {
            const post = await moepics.posts.get(id)
            if (post?.postID) return
        } else {
            console.log(`deleted: ${id}`)
        }
    }
}

const massUpdateSketch = async () => {
    const moepics = new Moepictures(process.env.MOEPICTURES_API_KEY!)

    let sketchFolder = process.env.FOLDER!
    
    const sketchFiles = fs.readdirSync(sketchFolder).filter((f) => !f.includes(".DS_Store"))
    .sort(new Intl.Collator(undefined, {numeric: true, sensitivity: "base"}).compare)

    for (const file of sketchFiles) {
        let id = file.split("_")[0]?.match(/\d+/)?.[0]!
        if (!id) continue
        if (Number(id) < 0) continue
        await moepics.posts.update(id, "style", "sketch")
        console.log(`${id}`)
    }
}

export default massDeletePosts