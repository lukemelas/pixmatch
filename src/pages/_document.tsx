import NextDocument, { Html, Head, Main, NextScript } from 'next/document'
import { ColorModeScript } from '@chakra-ui/react'

const pageTitle = "PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training"
const description = "We introduce a new loss term to enforce pixelwise consistency between the model's predictions on a target image and a perturbed version of the same image. In comparison to popular adversarial adaptation methods, our approach is simpler, easier to implement, and more memory-efficient during training. Experiments and extensive ablation studies demonstrate that our simple approach achieves remarkably strong results on two challenging synthetic-to-real benchmarks, GTA5-to-Cityscapes and SYNTHIA-to-Cityscapes."


export default class Document extends NextDocument {
  render() {
    return (
      <Html>
        <Head />
        {/* TODO: Set all these attributes */}
        <title>{pageTitle}</title>
        <meta name="description" content={description} />
        <meta property="og:title" content={pageTitle} key="ogtitle" />
        <meta property="og:description" content={description} key="ogdesc" />
        <meta property="og:site_name" content={pageTitle} key="ogsitename" />
        {/* <meta property="og:url" content={TODO} key="ogurl" /> */}
        {/* <meta name="twitter:card" content="{summary}" key="twcard" /> */}
        {/* <meta name="twitter:creator" content={twitterHandle} key="twhandle" /> */}
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta charSet="utf-8" />
        <body>
          {/* Make Color mode to persists when you refresh the page. */}
          <ColorModeScript />
          <Main />
          <NextScript />
        </body>
      </Html>
    )
  }
}
