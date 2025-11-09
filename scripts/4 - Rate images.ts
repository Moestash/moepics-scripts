import Moepictures from "moepics-api"
import fs from "fs"
import path from "path"

const rateImages = async () => {
    const moepics = new Moepictures(process.env.MOEPICTURES_API_KEY!)

    let cuteDir = path.join(process.env.FOLDER!, "cute")
    let sexyDir = path.join(process.env.FOLDER!, "sexy")
    let eroticDir = path.join(process.env.FOLDER!, "erotic")

    const cuteFiles = fs.readdirSync(cuteDir).filter((f) => !f.includes(".DS_Store"))
    .sort(new Intl.Collator(undefined, {numeric: true, sensitivity: "base"}).compare)
    for (const file of cuteFiles) {
         let id = file.split("_")[0]?.match(/\d+/)?.[0]!
         console.log(`${id}: cute`)
         const result = await moepics.posts.update(id, "rating", "cute")
         if (result?.toLowerCase()?.trim() !== "success") {
            const post = await moepics.posts.get(id)
            if (post?.postID) return
         }
    }

    const sexyFiles = fs.readdirSync(sexyDir).filter((f) => !f.includes(".DS_Store"))
    .sort(new Intl.Collator(undefined, {numeric: true, sensitivity: "base"}).compare)
    for (const file of sexyFiles) {
         let id = file.split("_")[0]?.match(/\d+/)?.[0]!
         console.log(`${id}: sexy`)
         const result = await moepics.posts.update(id, "rating", "sexy")
         if (result?.toLowerCase()?.trim() !== "success") {
            const post = await moepics.posts.get(id)
            if (post?.postID) return
         }
    }

    const eroticFiles = fs.readdirSync(eroticDir).filter((f) => !f.includes(".DS_Store"))
    .sort(new Intl.Collator(undefined, {numeric: true, sensitivity: "base"}).compare)
    for (const file of eroticFiles) {
         let id = file.split("_")[0]?.match(/\d+/)?.[0]!
         console.log(`${id}: erotic`)
         const result = await moepics.posts.update(id, "rating", "erotic")
         if (result?.toLowerCase()?.trim() !== "success") {
            const post = await moepics.posts.get(id)
            if (post?.postID) return
         }
    }
}

export default rateImages