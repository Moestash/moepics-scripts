import Moepictures from "moepics-api"

const addArtToolTags = async () => {
    const moepics = new Moepictures(process.env.MOEPICTURES_API_KEY!)

    const posts = await moepics.search.posts({query: "", type: "image", rating: "all+l", style: "all+s", sort: "reverse date", limit: 99999})

    let i = 0
    let skip = 0
    for (const post of posts) {
        i++
        if (i < skip) continue
        let drawingTools = post.drawingTools as string[]
        if (drawingTools?.length) {
            let ignored = ["ミリペン", "コピック", "筆", "シャープペンシル", "鉛筆", "エアブラシ", "サインペン",
            "透明水彩", "万年筆", "ボールペン", "お絵描き掲示板", "つけペン", "筆ペン", "色鉛筆", "水彩色鉛筆", "顔彩",
            "お絵描きチャット", "クーピーペンシル", "マジック", "パステル", "絵の具", "クレヨン", "アクリル", "Fireworks",
            "Expression"]
            for (const tool of drawingTools) {
                if (tool === "CLIP STUDIO PAINT") {
                    await moepics.posts.addTags(post.postID, ["clip-studio-paint"])
                } else if (tool === "SAI") {
                    await moepics.posts.addTags(post.postID, ["paint-tool-sai"])
                } else if (tool === "Photoshop") {
                    await moepics.posts.addTags(post.postID, ["photoshop"])
                } else if (tool === "Live2D") {
                    await moepics.posts.addTags(post.postID, ["live2d"])
                } else if (tool === "IllustStudio") {
                    await moepics.posts.addTags(post.postID, ["illust-studio"])
                } else if (tool === "PhotoStudio") {
                    await moepics.posts.addTags(post.postID, ["photo-studio"])
                } else if (tool === "ComicStudio") {
                    await moepics.posts.addTags(post.postID, ["comic-studio"])
                } else if (tool === "FireAlpaca") {
                    await moepics.posts.addTags(post.postID, ["fire-alpaca"])
                } else if (tool === "MediBang Paint") {
                    await moepics.posts.addTags(post.postID, ["medibang-paint"])
                } else if (tool === "MediBang Paint Pro") {
                    await moepics.posts.addTags(post.postID, ["medibang-paint"])
                } else if (tool === "Procreate") {
                    await moepics.posts.addTags(post.postID, ["procreate"])
                } else if (tool === "GIMP") {
                    await moepics.posts.addTags(post.postID, ["gimp"])
                } else if (tool === "Pixia") {
                    await moepics.posts.addTags(post.postID, ["pixia"])
                } else if (tool === "openCanvas") {
                    await moepics.posts.addTags(post.postID, ["open-canvas"])
                } else if (tool === "Illustrator") {
                    await moepics.posts.addTags(post.postID, ["illustrator"])
                } else if (tool === "Poser") {
                    await moepics.posts.addTags(post.postID, ["poser"])
                } else if (tool === "Blender") {
                    await moepics.posts.addTags(post.postID, ["blender"])
                } else if (tool === "MS_Paint") {
                    await moepics.posts.addTags(post.postID, ["ms-paint"])
                } else if (tool === "AzPainter2") {
                    await moepics.posts.addTags(post.postID, ["azpainter"])
                } else if (tool === "AzPainter") {
                    await moepics.posts.addTags(post.postID, ["azpainter"])
                } else if (tool === "Krita") {
                    await moepics.posts.addTags(post.postID, ["krita"])
                } else if (tool === "Aseprite") {
                    await moepics.posts.addTags(post.postID, ["aseprite"])
                } else if (tool === "AfterEffects") {
                    await moepics.posts.addTags(post.postID, ["after-effects"])
                } else if (tool === "ibisPaint") {
                    await moepics.posts.addTags(post.postID, ["ibis-paint"])
                } else if (tool === "ZBrush") {
                    await moepics.posts.addTags(post.postID, ["zbrush"])
                } else if (tool === "Maya") {
                    await moepics.posts.addTags(post.postID, ["maya"])
                } else if (tool === "3dsMax") {
                    await moepics.posts.addTags(post.postID, ["3ds-max"])
                } else if (tool === "CINEMA4D") {
                    await moepics.posts.addTags(post.postID, ["cinema4d"])
                } else if (tool === "Inkscape") {
                    await moepics.posts.addTags(post.postID, ["inkscape"])
                } else if (tool === "Inkscape") {
                    await moepics.posts.addTags(post.postID, ["inkscape"])
                } else if (tool === "AzDrawing2") {
                    await moepics.posts.addTags(post.postID, ["azdrawing"])
                } else if (tool === "AzDrawing") {
                    await moepics.posts.addTags(post.postID, ["azdrawing"])
                } else if (tool === "pixiv Sketch") {
                    await moepics.posts.addTags(post.postID, ["pixiv-sketch"])
                } else if (tool === "VRoid Studio") {
                    await moepics.posts.addTags(post.postID, ["vroid-studio"])
                } else if (tool === "RETAS STUDIO") {
                    await moepics.posts.addTags(post.postID, ["retas-studio"])
                } else if (tool === "drawr") {
                    await moepics.posts.addTags(post.postID, ["drawr"])
                } else if (tool === "Paintgraphic") {
                    await moepics.posts.addTags(post.postID, ["paintgraphic"])
                } else if (tool === "COMICWORKS") {
                    await moepics.posts.addTags(post.postID, ["comicworks"])
                } else if (tool === "CLIP PAINT Lab") {
                    await moepics.posts.addTags(post.postID, ["clip-paint-lab"])
                } else if (tool === "SketchBookPro") {
                    await moepics.posts.addTags(post.postID, ["sketchbook-pro"])
                } else if (tool === "PaintShopPro") {
                    await moepics.posts.addTags(post.postID, ["paintshop-pro"])
                } else if (tool === "CGillust") {
                    await moepics.posts.addTags(post.postID, ["cgillust"])
                } else if (tool === "Metasequoia") {
                    await moepics.posts.addTags(post.postID, ["metasequoia"])
                } else if (tool === "Metasequoia") {
                    await moepics.posts.addTags(post.postID, ["metasequoia"])
                } else if (tool === "Shade") {
                    await moepics.posts.addTags(post.postID, ["shade3d"])
                } else if (tool === "Flash") {
                    await moepics.posts.addTags(post.postID, ["flash"])
                } else if (tool === "Painter") {
                    await moepics.posts.addTags(post.postID, ["corel-painter"])
                } else if (ignored.includes(tool)) {
                    // ignore
                } else {
                    console.log(`Unknown tool: ${tool}`)
                    return
                }
            }
            console.log(`${i} -> ${post.postID}`)
        }
    }
}

export default addArtToolTags