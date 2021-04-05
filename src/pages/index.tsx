import { Link as ChakraLink, Image as ChakraImage, Text, Code, ListItem, Heading, UnorderedList, Box, } from '@chakra-ui/react'
import { Hero } from 'components/Hero'
import { Container } from 'components/Container'
import NextLink from 'next/link'
import { DarkModeSwitch } from 'components/DarkModeSwitch'
import { LinksRow } from 'components/LinksRow'
import { Footer } from 'components/Footer'

const Index = () => (
  <Container>

    {/* Edit author info */}
    <Hero title="PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training" />

    {/* TODO: Add paper/github links here */}
    <LinksRow />

    {/* TODO: Add video */}
    <Container w="90vw" h="50.6vw" maxW="700px" maxH="393px" mb="3rem">
      <iframe
        width="100%" height="100%"
        src="https://www.youtube.com/embed/ScMzIvxBSi4"
        title="Video"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen>
      </iframe>
    </Container>

    <Container w="100%" maxW="44rem" alignItems="left" pl="1rem" pr="1rem">

      {/* Abstract */}
      <Heading fontSize="2xl" pb="1rem">Abstract</Heading>
      <Text pb="2rem">
        Unsupervised domain adaptation is a promising technique for semantic segmentation and other computer vision tasks for which large-scale data annotation is costly and time-consuming. In semantic segmentation, it is attractive to train models on annotated images from a simulated (source) domain and deploy them on real (target) domains. In this work, we present a novel framework for unsupervised domain adaptation based on the notion of target-domain consistency training. Intuitively, our work is based on the idea that in order to perform well on the target domain, a model’s output should be consistent with respect to small perturbations of inputs in the target domain. Specifically, we introduce a new loss term to enforce pixelwise consistency between the model's predictions on a target image and a perturbed version of the same image. In comparison to popular adversarial adaptation methods, our approach is simpler, easier to implement, and more memory-efficient during training. Experiments and extensive ablation studies demonstrate that our simple approach achieves remarkably strong results on two challenging synthetic-to-real benchmarks, GTA5-to-Cityscapes and SYNTHIA-to-Cityscapes.
      </Text>

      {/* Example */}
      <Heading fontSize="2xl" pb="1rem">Approach</Heading>
      <ChakraImage src={`${process.env.BASE_PATH || ""}/images/diagram.jpg`} />
      <Text align="center" pt="0.5rem" pb="0.5rem" fontSize="small">Our proposed pixelwise consistency training approach.</Text>

      {/* Another Section */}
      <Heading fontSize="2xl" pt="2rem" pb="1rem">Examples</Heading>
      <ChakraImage src={`${process.env.BASE_PATH || ""}/images/example-synthia.jpg`} />
      <Text align="center" pt="0.5rem" pb="0.5rem" fontSize="small">Qualitative results on SYNTHIA-to-Cityscapes</Text>


      {/* Citation */}
      <Heading fontSize="2xl" pt="2rem" pb="1rem">Citation</Heading>
      <Box w="100%" overflow="scroll">
        <Code p="0.5rem" borderRadius="5px" w="max-content">
          {/* w="150%"> */}
          @inproceedings&#123; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;yu2021plenoctrees, <br />
          &nbsp;&nbsp;&nbsp;&nbsp;title=&#123;PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;author=&#123;Luke Melas-Kyriazi and Arjun K. Manrai&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;year=&#123;2021&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;booktitle=&#123;CVPR&#125; <br />
      &#125;
      </Code>
      </Box>

      {/* Related Work */}
      <Heading fontSize="2xl" pt="2rem" pb="1rem">Related Work</Heading>
      <UnorderedList>
        <ListItem>
          <Text color="blue">
            <NextLink href="#" passHref={true}>
              First paper
            </NextLink>
          </Text>
        </ListItem>
        <ListItem>
          <Text color="blue">
            <NextLink href="#" passHref={true}>
              Second paper
            </NextLink>
          </Text>
        </ListItem>
      </UnorderedList>

      {/* Acknowledgements */}
      {/* <Heading fontSize="2xl" pt="2rem" pb="1rem">Acknowledgements</Heading>
      <Text >
        We thank xyz for abc...
      </Text> */}
    </Container>

    <DarkModeSwitch />
    <Footer>
      <Text></Text>
    </Footer>
  </Container >
)

export default Index
